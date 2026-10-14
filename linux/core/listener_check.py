"""Check bindability without confusing closed POSIX connections with listeners."""
import os, socket, time

def probe_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name == 'nt' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, int(port)))

def wait_ports(cfg, timeout=8):
    ports=[(cfg.get('host','127.0.0.1'),cfg.get('port',8298))]
    ports += [('127.0.0.1',cfg.get(k,v)) for k,v in [('image_port',8299)]]
    end=time.monotonic()+timeout
    while True:
        blocked=[]
        for host, port in ports:
            try: probe_port(host,port)
            except OSError: blocked.append(port)
        if not blocked: return
        if time.monotonic()>=end:
            raise RuntimeError('Listener still active on ports '+', '.join(map(str,blocked))+'. No unrelated process has been stopped.')
        time.sleep(.2)
