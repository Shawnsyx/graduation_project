#!/usr/bin/python
# -*- coding:utf8 -*-

import tornado.ioloop
import tornado.web
from dialog import Dailog
from config import make_config

config = make_config()
dialog = Dailog(config)

class WechatHandler(tornado.web.RequestHandler):

    def get(self):
        question = self.get_argument("question")
        question = question.strip().replace('"', '').replace("'", '')
        answer = dialog.answer(question)

        self.write(answer)

settings = {
    'template_path': 'template',
    'static_path': 'static',
}

application = tornado.web.Application(
    [(r"/", WechatHandler), (r"/index", WechatHandler), (r'/ajax/answer', WechatHandler)],
    **settings
)

if __name__ == '__main__':
    application.listen(8081)
    tornado.ioloop.IOLoop.instance().start()