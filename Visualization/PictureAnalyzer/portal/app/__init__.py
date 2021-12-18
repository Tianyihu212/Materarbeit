#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask
from app.view.biz_view import IndexView, FileAnalyseView, UrlAnalyseView


_app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
_app.add_url_rule('/', view_func=IndexView.as_view(name='index'), endpoint='index',  methods=['GET'])
_app.add_url_rule('/file_analyse', view_func=FileAnalyseView.as_view(name='file_analyse'), endpoint='file_analyse', methods=['POST'])
_app.add_url_rule('/url_analyse', view_func=UrlAnalyseView.as_view(name='url_analyse'), endpoint='url_analyse', methods=['POST'])

if __name__ == "__main__":
    _app.run(host='0.0.0.0',debug=True)
