"""Per-user self-signed local certificate. No CA trust-store changes."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import ipaddress, os, tempfile
from core.private_files import private_directory,protect

def generate_certificate(folder, lan_address=None):
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    folder=private_directory(folder)
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"Zetalvx Image Lab - Qwen 2.1 Edition")])
    names=[x509.DNSName("localhost"),x509.IPAddress(ipaddress.ip_address("127.0.0.1")),x509.IPAddress(ipaddress.ip_address("::1"))]
    if lan_address:names.append(x509.IPAddress(ipaddress.ip_address(lan_address)))
    now=datetime.now(timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=5))
        .not_valid_after(now+timedelta(days=365)).add_extension(x509.SubjectAlternativeName(names),critical=False)
        .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True).sign(key,hashes.SHA256()))
    outputs={"key.pem":key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()),"cert.pem":cert.public_bytes(serialization.Encoding.PEM)}
    for filename,data in outputs.items():
        fd,tmp=tempfile.mkstemp(dir=folder,prefix=filename+".")
        try:
            with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
            protect(tmp);os.replace(tmp,folder/filename)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
