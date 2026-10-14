"""HTTPS-only downloader transport: public IP pinning and origin-bound credentials.
No proxy env, cookies, netrc, pickle loading, or cross-origin token forwarding.
"""
from __future__ import annotations
import ipaddress, json, os, re, socket, threading
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, urljoin, parse_qsl, unquote
import urllib3
import requests
from core.private_files import private_directory, protect, atomic_private_json
from core.safe_paths import is_link

class DownloadError(ValueError): pass
class Cancelled(DownloadError): pass

SENSITIVE_QUERY = {'token','api_key','apikey','access_token','authorization','auth','key','password'}

def parse_url(url: str):
    if not isinstance(url,str) or not url or len(url)>8192 or any(ord(c)<33 for c in url) or '\\' in url:
        raise DownloadError('Incolla un URL HTTPS valido, senza spazi o credenziali.')
    try:
        p=urlsplit(url)
        port=p.port
        host=(p.hostname or '').encode('idna').decode('ascii').lower()
    except (ValueError,UnicodeError):raise DownloadError('URL non valido.') from None
    if p.scheme.lower()!='https' or not host or port not in (None,443) or p.username or p.password:
        raise DownloadError('Sono ammessi solo URL HTTPS pubblici sulla porta 443, senza utente/password.')
    if p.fragment:raise DownloadError('Rimuovi il frammento (#...) dal link del file.')
    return p,host

def clean_user_url(url):
    p,host=parse_url(str(url).strip())
    for k,_ in parse_qsl(p.query,keep_blank_values=True):
        if k.lower() in SENSITIVE_QUERY:
            raise DownloadError('Non mettere il token nel link. Salvalo in Modelli → Account e rimuovilo dall’URL.')
    return str(url).strip()

def public_addresses(host):
    if host.endswith(('.local','.localhost','.internal')) or host=='localhost':
        raise DownloadError('Download da rete locale o indirizzi privati non consentito. Usa i percorsi locali.')
    try: infos=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
    except OSError:raise DownloadError('Host non raggiungibile: verifica DNS e connessione Internet.') from None
    addresses=[]
    for info in infos:
        try: ip=ipaddress.ip_address(info[4][0])
        except ValueError:raise DownloadError('Indirizzo di rete non valido.') from None
        if not ip.is_global or ip.is_multicast or (getattr(ip,'ipv4_mapped',None) and not ip.ipv4_mapped.is_global):
            raise DownloadError('L’URL punta a un indirizzo privato/non pubblico: download rifiutato.')
        if str(ip) not in addresses:addresses.append(str(ip))
    if not addresses:raise DownloadError('Nessun indirizzo pubblico trovato.')
    return addresses

def safe_source(url):
    """Remove all query/fragment (including signed CDN links) from public records."""
    p=urlsplit(url)
    return urlunsplit((p.scheme,p.netloc,p.path,'',''))

def safe_name(name):
    name=unquote(str(name or ''))
    name=Path(name.replace('\\','/')).name
    name=re.sub(r'[^\w.\- ]','_',name,flags=re.UNICODE).strip(' .')
    if not name or len(name)>180 or name.startswith('.') or not name.lower().endswith(('.safetensors','.gguf')):
        raise DownloadError('Choose a .safetensors or transformer .gguf file. Pickle/bin checkpoints and archives are not accepted here.')
    return name

class TokenStore:
    """Local per-app secret store. Plaintext fallback, explicit 0700/0600 permissions.
    It deliberately never uses HF global login cache or returns token suffixes to JS.
    """
    PROVIDERS=('huggingface','civitai')
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.RLock()
    def _path(self,provider):
        if provider not in self.PROVIDERS:raise DownloadError('Piattaforma non supportata.')
        return self.root/(provider+'.json')
    def _prepare(self):
        if is_link(self.root):raise DownloadError('La cartella credenziali non può essere un link simbolico.')
        private_directory(self.root)
    def get(self,provider):
        with self.lock:
            p=self._path(provider)
            if not p.exists():return ''
            if is_link(p):raise DownloadError('Archivio credenziali non valido.')
            self._prepare();protect(p)
            try:return str(json.loads(p.read_text(encoding='utf-8')).get('token') or '')
            except (ValueError,OSError):raise DownloadError('Impossibile leggere il token salvato. Ricollega l’account.') from None
    def put(self,provider,token,account=''):
        import tempfile
        if not isinstance(token,str) or not 8<=len(token)<=2048 or any(c.isspace() or not 33<=ord(c)<=126 for c in token):
            raise DownloadError('Token non valido: incollalo senza spazi.')
        with self.lock:
            p=self._path(provider);self._prepare()
            if is_link(p):raise DownloadError('Archivio credenziali non valido.')
            atomic_private_json(p,{'token':token,'account':str(account)[:120]})
    def delete(self,provider):
        with self.lock:
            if self.root.exists():self._prepare()
            self._path(provider).unlink(missing_ok=True)
    def public(self):
        result={}
        for provider in self.PROVIDERS:
            p=self._path(provider);account=''
            if p.is_file() and not is_link(p):
                self._prepare();protect(p)
                try:account=str(json.loads(p.read_text(encoding='utf-8')).get('account') or '')[:120]
                except (ValueError,OSError):pass
            result[provider]={'stored':bool(p.is_file() and not p.is_symlink()),'account':account,
                              'storage':('File locale con ACL Windows, non cifrato' if os.name=='nt' else 'File locale protetto (0600), non cifrato')}
        return result

class Response:
    def __init__(self,raw,pool,url):
        self.raw=raw;self.pool=pool;self.url=url;self.status=raw.status;self.headers=raw.headers
    def chunks(self,size=1024*1024):
        yield from self.raw.stream(size,decode_content=False)
    def small(self,limit=8*1024*1024):
        data=bytearray()
        for part in self.chunks(65536):
            data.extend(part)
            if len(data)>limit:raise DownloadError('Risposta remota troppo grande.')
        return bytes(data)
    def close(self):self.raw.close();self.pool.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

class PublicHTTPS:
    def __init__(self,tokens):self.tokens=tokens
    @staticmethod
    def provider(host):
        return 'huggingface' if host=='huggingface.co' else 'civitai' if host in ('civitai.com','www.civitai.com') else None
    def open(self,url, *, token_override=None, cancel=None):
        # Credentials only for the INITIAL trusted origin, never reintroduced after crossing origins.
        _,origin=parse_url(url);current=url;auth_allowed=True
        for _ in range(8):
            if cancel and cancel.is_set():raise Cancelled('Download annullato.')
            p,host=parse_url(current)
            from core.model_policy import external_reason
            reason=external_reason(current)
            if reason:raise DownloadError(reason)
            auth_allowed=auth_allowed and host==origin
            addresses=public_addresses(host);last=None
            for address in addresses:
                pool=urllib3.HTTPSConnectionPool(address,443,server_hostname=host,assert_hostname=host,
                      cert_reqs='CERT_REQUIRED',ca_certs=requests.certs.where(),
                      timeout=urllib3.Timeout(connect=12,read=30),retries=False,maxsize=1)
                headers={'Host':host,'User-Agent':'ImageLabQwen21/0.1.0.9','Accept-Encoding':'identity'}
                provider=self.provider(host)
                if auth_allowed and provider:
                    token=token_override if token_override is not None else self.tokens.get(provider)
                    if token:headers['Authorization']='Bearer '+token
                try:
                    raw=pool.urlopen('GET',urlunsplit(('','',p.path or '/',p.query,'')),headers=headers,
                                     redirect=False,preload_content=False,retries=False,assert_same_host=False)
                    break
                except (urllib3.exceptions.HTTPError,OSError) as e:
                    # Never surface raw network exceptions containing URLs or authorization.
                    last=e;pool.close()
            else:raise DownloadError('Connessione HTTPS non riuscita. Verifica rete/certificati e riprova.') from None
            if raw.status in (301,302,303,307,308):
                location=raw.headers.get('Location','');raw.close();pool.close()
                if not location:raise DownloadError('Redirect privo di destinazione.')
                current=urljoin(current,location)
                continue
            if raw.status not in (200,):
                status=raw.status;raw.close();pool.close()
                if status in (401,403):raise DownloadError('Accesso negato (401/403). Collega il tuo token e accetta/richiedi accesso sulla pagina ufficiale; nessun bypass.')
                if status==404:raise DownloadError('File non trovato (404). Controlla il link e la versione.')
                if status==429:raise DownloadError('Limite richieste raggiunto (429). Attendi e riprova.')
                raise DownloadError(f'Il server ha risposto HTTP {status}.')
            return Response(raw,pool,current)
        raise DownloadError('Troppi redirect: download interrotto.')
    def json(self,url,token_override=None):
        with self.open(url,token_override=token_override) as r:
            try:return json.loads(r.small())
            except (ValueError,UnicodeError):raise DownloadError('La piattaforma non ha restituito metadati JSON validi.') from None
