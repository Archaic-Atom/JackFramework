# -*- coding: utf-8 -*-
import os
import django
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from JackFramework.SysBasic.log_handler import LogHandler as log
from JackFramework.Core.Mode.web_proc import WebProc


class Views(object):
    UPLOADS_FOLDER = 'uploads'
    RESULT_FOLDER = 'results'

    """docstring for Views"""

    def __init__(self, args: object):
        super().__init__()
        self.__args = args

    @staticmethod
    def get_abs_path(root_folder: str, file_list: list) -> str:
        return [os.path.join(
            root_folder, os.path.basename(file_name)) for file_name in file_list]

    @staticmethod
    def create_folder() -> tuple:
        result_folder = os.path.join(settings.MEDIA_ROOT, Views.RESULT_FOLDER)
        uploads_folder = os.path.join(settings.MEDIA_ROOT, Views.UPLOADS_FOLDER)

        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(uploads_folder, exist_ok=True)

        log.info(f'The uploaded files in {uploads_folder}\nThe results files in {result_folder}')
        return result_folder, uploads_folder

    @staticmethod
    def receive_files(files: django.http.HttpRequest,
                      uploads_folder: str) -> list:
        fs = FileSystemStorage(uploads_folder)
        return [os.path.join(
            settings.MEDIA_URL, Views.UPLOADS_FOLDER, fs.save(f.name, f)) for f in files]

    @staticmethod
    def web_proc(files_path: list, uploads_folder: str, result_folder: str) -> list:
        web_proc = WebProc.get_web_object()
        log.info(f'The web handler is {web_proc}')
        msg = Views.get_abs_path(uploads_folder, files_path)
        msg.append(result_folder)
        return web_proc.data_handler(msg)

    @staticmethod
    def generate_results_path(res_files: list) -> list:
        return [os.path.join(
            settings.MEDIA_URL, Views.RESULT_FOLDER, item) for item in res_files]

    @staticmethod
    def run(request: django.http.HttpRequest) -> django.shortcuts:
        files_path, res_files_path = None, None
        if request.method == 'POST':
            try:
                files = request.FILES.getlist('images')
                result_folder, uploads_folder = Views.create_folder()
                files_path = Views.receive_files(files, uploads_folder)
                res_files = Views.web_proc(files_path, uploads_folder, result_folder)

                if res_files is not False:
                    res_files_path = Views.generate_results_path(res_files)
            except Exception as e:
                log.error(f"An error occurred during processing: {str(e)}")

        return render(request, 'webapp/index.html',
                      {'images': files_path, 'res_images': res_files_path})
