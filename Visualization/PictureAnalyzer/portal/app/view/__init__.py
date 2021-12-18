import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(MAIN_DIR, "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")
IMG_DIR = os.path.join(STATIC_DIR, "img")