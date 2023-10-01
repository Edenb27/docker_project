import requests
import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


def yolo5_request(s3_img_path):
    yolo5_api_url = "http://yolo5-app:8081/predict"
    response = requests.post(f"{yolo5_api_url}?imgName={s3_img_path}")
    summary = response.json()
    labelsDic = {}
    for label in summary['labels']:
        try:
            labelsDic[label['class']] += 1
        except:
            labelsDic.update({label['class']: 1})

    predicted_path = summary['predicted_img_path']
    logger.info(f'predicted_path:   {predicted_path}')
    return labelsDic, predicted_path


class ObjectDetectionBot(Bot):

    def handle_message(self, msg):
        # logger.info(f'Incoming message: {msg}')
        if msg.get('text'):
            hello = f'\nHello and welcome to Eden Predict-Bot \n' \
                    f'\n Please send image to prediction'
            self.send_text(msg['chat']['id'], hello)

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            BUCKET_NAME = 'edenb27-docker'
            img_name = f'images/{photo_path}'
            client = boto3.client('s3')
            client.upload_file(photo_path, BUCKET_NAME, img_name)
            summary_dic, s3_pred_path = yolo5_request(img_name)
            summary_label = ''
            for key in summary_dic.keys():
                summary_label = summary_label + key + ": " + summary_dic[key].__str__() + " "

            logger.info(f'summary_label:    {summary_label}')
            logger.info(f'summary_dic:  {summary_dic}')
            filename = photo_path.split('/')[-1]
            #pred_img_name = f'predicted_{filename}'
            # s3_pred_path = '/'.join(img_name.split('/')[:-1]) + f'/{pred_img_name}'
            local_path = f'images/pred/{filename}'
            os.makedirs('images/pred/', exist_ok=True)
            client.download_file(BUCKET_NAME, s3_pred_path, local_path)
            self.send_photo(msg['chat']['id'], local_path)
            self.send_text(msg['chat']['id'], f'prediction: {summary_label}')

            # TODO upload the photo to S3
            # TODO send a request to the `yolo5` service for prediction
            # TODO send results to the Telegram end-user
