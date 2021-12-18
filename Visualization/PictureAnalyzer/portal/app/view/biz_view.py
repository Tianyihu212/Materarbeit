import os
import json
from urllib.request import urlretrieve
from flask import views, render_template, request
from app.view import IMG_DIR
from app.main.util import get_local_img_b64, get_mat_img_b64_list, get_img_mat_list
from app.main import test

class IndexView(views.MethodView):
    template_name = 'index.html'

    def get(self):
        return render_template(self.template_name)


class FileAnalyseView(views.MethodView):
    template_name = 'result.html'

    def post(self):
        f = request.files.get('file')
        img_file = os.path.join(IMG_DIR, "src.png")
        f.save(img_file)
        src_file = get_local_img_b64(img_file, 'png')

        # img_mat_list = get_img_mat_list()
        img_mat_list = test.query(img_file)
        dst_files = get_mat_img_b64_list(img_mat_list)

        return render_template(self.template_name, srcFile=src_file, dstFiles=dst_files)


class UrlAnalyseView(views.MethodView):
    template_name = 'result.html'

    def post(self):
        img_url = json.loads(request.get_data())['url']
        img_file = os.path.join(IMG_DIR, "src.png")
        urlretrieve(img_url, img_file)
        src_file = get_local_img_b64(img_file, 'png')

        # img_mat_list = get_img_mat_list()
        img_mat_list = test.query(img_file)
        dst_files = get_mat_img_b64_list(img_mat_list)

        return render_template(self.template_name, srcFile=src_file, dstFiles=dst_files)