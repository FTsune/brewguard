import os

bind = "0.0.0.0:" + os.environ.get("PORT", "10000")
workers = 1  # For ML models, often 1 worker is better
timeout = 120  # Longer timeout for processing images
threads = 2
worker_class = "sync"
loglevel = "info"
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stdout