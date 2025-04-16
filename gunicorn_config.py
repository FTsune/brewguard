import os

bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 1
timeout = 300
threads = 2
worker_class = "sync"
loglevel = "info"
accesslog = "-"
errorlog = "-"
preload_app = False  # Important: Set to False for background loading