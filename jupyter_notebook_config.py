import os

c = get_config()

c.ServerApp.jpserver_extensions = {
    'js9server.js9ext': True,
}

vnc_socket = os.path.join(os.getenv('HOME'), '.vnc', 'socket')
xstartup = 'dbus-launch xfce4-session'

noVNC_version = '1.1.0'

c.ServerProxy.servers = {
    'PPANDA': {
        'command': [
            'streamlit',
            'run',
            'ppanda_streamlit.py',
            '--server.port', '8501',
            '--browser.serverAddress', '0.0.0.0',
        ],
        'port': 8501,
        'timeout': 60,
        'launcher_entry': {
            'enabled': True,
            'title': 'PPANDA'
        }
    },
}

print("\033[31mconfiguring!\033[0m")
