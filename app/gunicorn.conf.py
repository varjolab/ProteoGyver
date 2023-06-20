import multiprocessing

bind = "0.0.0.0:8050"

workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'async'
worker_connections = 1000
timeout = 1200
spew = False