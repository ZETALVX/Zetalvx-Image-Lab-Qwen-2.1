"""WSGI transport; same Flask application, port, TLS and processes.
Cheroot is imported lazily and installed only in runtime/app. No HTTP fallback.
"""
from __future__ import annotations
import signal, ssl, threading

def make_server(app,host,port,*,tls=None):
    from cheroot.wsgi import Server
    from cheroot.ssl.builtin import BuiltinSSLAdapter
    from cheroot.server import HTTPConnection, HTTPRequest
    from core.http_body_cleanup import finish_unread_body

    class StudioRequest(HTTPRequest):
        def send_headers(self):
            # Preserve the WSGI response (including denial status) while making
            # small early-rejected TLS POSTs safe with Connection: close.
            finish_unread_body(self)
            return super().send_headers()

    class StudioConnection(HTTPConnection):
        RequestHandlerClass = StudioRequest
    server=Server((host,int(port)),app,numthreads=12,max=24,request_queue_size=64,
                  timeout=120,shutdown_timeout=5,accepted_queue_size=128)
    server.ConnectionClass = StudioConnection
    server.max_request_header_size=64*1024
    # Endpoint-specific Flask limits continue to control uploads/chunks.
    server.max_request_body_size=0
    if tls:
        adapter=BuiltinSSLAdapter(*tls)
        adapter.context.minimum_version=ssl.TLSVersion.TLSv1_2
        server.ssl_adapter=adapter
    return server

def serve(app,host,port,*,tls=None):
    server=make_server(app,host,port,tls=tls)
    previous={}
    def stop(*_):
        threading.Thread(target=server.stop,daemon=True,name='web-shutdown').start()
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM,signal.SIGINT):
            previous[sig]=signal.getsignal(sig);signal.signal(sig,stop)
    print('Zetalvx Image Lab web: '+('https' if tls else 'http')+'://'+str(host)+':'+str(port)+' (Cheroot)',flush=True)
    try:server.start()
    finally:
        server.stop()
        for sig,handler in previous.items():signal.signal(sig,handler)
