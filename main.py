import sys
import subprocess
import os
from threading import Thread
from time import sleep
import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox, \
    QWidget, QFileDialog, QPushButton

from CNN.cnn import CNN
from pytorch.GoogleNet import GoogleNet
from pytorch.AlexNet import Alexnet
from pytorch.VGG import VGG


class NN_UI(QMainWindow):
    def __init__(self):
        super(NN_UI, self).__init__()

        self.setWindowTitle('Подбор архитектуры нейронной сети')
        self.setFixedSize(800, 400)

        self.layoutV = QVBoxLayout()
        self.layoutH1 = QHBoxLayout()
        self.layoutH2 = QHBoxLayout()
        self.layoutH3 = QHBoxLayout()
        self.layoutH4 = QHBoxLayout()

        self.layoutV.addLayout(self.layoutH1)

        self.text_model = QLabel('Выберите модель нейронной сети:')
        self.comboBox = QComboBox()
        self.comboBox.addItems(['CNN чтение текста', 'YOLO', 'VGG', 'GoogLeNet', 'AlexNet'])
        self.layoutH1.addWidget(self.text_model)
        self.layoutH1.addWidget(self.comboBox)

        self.text_train = QLabel('Выберите путь к данным для обучения:')
        self.btn_train_file = QPushButton('выбрать файл')
        self.btn_train_dir = QPushButton('выбрать папку')
        self.btn_train_file.clicked.connect(self.dialog_train_file)
        self.btn_train_dir.clicked.connect(self.dialog_train_dir)
        self.layoutH2.addWidget(self.text_train)
        self.layoutH2.addWidget(self.btn_train_file)
        self.layoutH2.addWidget(self.btn_train_dir)
        self.layoutV.addLayout(self.layoutH2)

        self.lbl_train = QLabel()

        self.layoutV.addWidget(self.lbl_train)

        self.text_test = QLabel('Выберите путь к данным для тестирования:')
        self.btn_test_file = QPushButton('выбрать файл')
        self.btn_test_dir = QPushButton('выбрать папку')
        self.btn_test_file.clicked.connect(self.dialog_test_file)
        self.btn_test_dir.clicked.connect(self.dialog_test_dir)
        self.layoutH3.addWidget(self.text_test)
        self.layoutH3.addWidget(self.btn_test_file)
        self.layoutH3.addWidget(self.btn_test_dir)
        self.layoutV.addLayout(self.layoutH3)
        self.lbl_test = QLabel()
        self.layoutV.addWidget(self.lbl_test)

        self.text_work = QLabel('Выберите путь к данным для обработки:')
        self.btn_work_file = QPushButton('выбрать файл')
        self.btn_work_dir = QPushButton('выбрать папку')
        self.btn_work_file.clicked.connect(self.dialog_work_file)
        self.btn_work_dir.clicked.connect(self.dialog_work_dir)
        self.layoutH4.addWidget(self.text_work)
        self.layoutH4.addWidget(self.btn_work_file)
        self.layoutH4.addWidget(self.btn_work_dir)
        self.layoutV.addLayout(self.layoutH4)
        self.lbl_work = QLabel()
        self.layoutV.addWidget(self.lbl_work)

        self.btn_start = QPushButton('Запустить')
        self.btn_start.clicked.connect(self.start)
        self.layoutV.addWidget(self.btn_start)

        # ----------------------------
        self.gbox = QGroupBox('Результаты')
        self.results_layout = QHBoxLayout()
        self.lbl_results = QLabel('Результаты работы:\n')
        self.results_layout.addWidget(self.lbl_results)
        self.gbox.setLayout(self.results_layout)
        self.layoutV.addWidget(self.gbox)
        # ----------------------------

        widget = QWidget()
        widget.setLayout(self.layoutV)
        self.setCentralWidget(widget)

    def dialog_train_file(self):
        self.dialog_train = QFileDialog()
        self.lbl_train.setText(self.dialog_train.getOpenFileName()[0])

    def dialog_train_dir(self):
        self.dialog_train = QFileDialog()
        self.lbl_train.setText(self.dialog_train.getExistingDirectory())

    def dialog_test_file(self):
        self.dialog_test = QFileDialog()
        self.lbl_test.setText(self.dialog_test.getOpenFileName()[0])

    def dialog_test_dir(self):
        self.dialog_test = QFileDialog()
        self.lbl_test.setText(self.dialog_test.getExistingDirectory())

    def dialog_work_file(self):
        self.dialog_work = QFileDialog()
        # print(self.dialog_work.getOpenFileName()[0])
        self.lbl_work.setText(self.dialog_work.getOpenFileName()[0])

    def dialog_work_dir(self):
        self.dialog_work = QFileDialog()
        # print(self.dialog_work.getOpenFileName()[0])
        self.lbl_work.setText(self.dialog_work.getExistingDirectory())

    def train_output(self, text):
        self.lbl_results.setText(text)
        sleep(1)

    def start(self):
        if self.comboBox.currentText() == 'YOLO':
            if self.lbl_train.text():
                self.lbl_results.setText('Модель обучается')
                train_path = self.lbl_train.text()
                command = 'python yolov5/train.py --data ' + train_path + ' --weights yolov5s.pt --img 640'
                process = subprocess.check_output(command, shell=True)
                self.lbl_results.setText('Модель обучена' + process)

            if self.lbl_test.text():
                self.lbl_results.setText('Выполняется тестирование')
                test_path = self.lbl_test.text()
                command = 'python yolov5/val.py --weights yolov5s.pt --data ' + test_path + ' --img 640'
                process = subprocess.check_output(command, shell=True)
                self.lbl_results.setText('Результат тестирования:' + process)

            if self.lbl_work.text():
                self.lbl_results.setText('Модель выполняет обработку')
                detect_path = self.lbl_work.text()
                command = 'python yolov5/detect.py --weights yolov5s.pt --source ' + detect_path
                process = subprocess.check_output(command, shell=True)
                print(process)
                self.lbl_results.setText('Файл сохранен по пути yolov5/runs/detect')

            if self.lbl_work.text() is None and self.lbl_test.text() is None and self.lbl_train.text() is None:
                self.lbl_results.setText('Укажите пути к файлам для обучения и/или обработки')

        if self.comboBox.currentText() == 'CNN чтение текста':
            if self.lbl_train.text():
                self.lbl_results.setText('Выполняется обучение нейронной сети')
                cnn = CNN()
                cnn.main(path1=self.lbl_train.text())
                self.lbl_results.setText(cnn.results)

            if self.lbl_test.text():
                self.lbl_results.setText('Выполняется обучение нейронной сети')
                cnn = CNN()
                cnn.main(path1=self.lbl_test.text())
                self.lbl_results.setText('Точность: ' + cnn.val)

            if self.lbl_work.text():
                cnn = CNN()
                cnn.main(path2=str(self.lbl_work.text()))
                self.lbl_results.setText('Текст: ' + cnn.results)

            if self.lbl_work.text() is None and self.lbl_test.text() is None and self.lbl_train.text() is None:
                self.lbl_results.setText('Укажите пути к файлам для обучения и/или тестирования')

        # -----------VGG------------------
        if self.comboBox.currentText() == 'VGG':
            if self.lbl_train.text() and self.lbl_test.text():
                self.lbl_results.setText('Выполняется обучение нейронной сети')
                parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
                parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
                parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
                parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
                parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
                parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                                    help='whether cuda is in use')
                args = parser.parse_args()

                vgg = VGG(args)
                vgg.run()
                self.lbl_results.setText(vgg.result)

            if self.lbl_work.text():
                self.lbl_results.setText("К сожалению данный функционал пока не поддерживается, попробуйте YOLO")

            if (self.lbl_work.text() is None or self.lbl_test.text()) is None or self.lbl_train.text() is None:
                self.lbl_results.setText('Укажите пути к файлам для обучения и/или тестирования')

        # -----------googlenet------------------------
        if self.comboBox.currentText() == 'GoogLeNet':
            if self.lbl_train.text() and self.lbl_test.text():
                self.lbl_results.setText('Выполняется обучение нейронной сети')
                parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
                parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
                parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
                parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
                parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
                parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                                    help='whether cuda is in use')
                args = parser.parse_args()
                print(self.lbl_train.text())
                print(self.lbl_test.text())

                googlenet = GoogleNet(args, train_path=self.lbl_train.text(), test_path=self.lbl_test.text())
                googlenet.run()
                self.lbl_results.setText(googlenet.result)

            if self.lbl_work.text():
                self.lbl_results.setText("К сожалению данный функционал пока не поддерживается, попробуйте YOLO")

            if (self.lbl_work.text() is None or self.lbl_test.text()) is None or self.lbl_train.text() is None:
                self.lbl_results.setText('Укажите пути к файлам для обучения и/или тестирования')

        # ------------------------------AlexNet------------------------
        if self.comboBox.currentText() == 'AlexNet':
            if self.lbl_train.text() and self.lbl_test.text():
                self.lbl_results.setText('Выполняется обучение нейронной сети')
                parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
                parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
                parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
                parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
                parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
                parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                                    help='whether cuda is in use')
                args = parser.parse_args()

                alex = Alexnet(args)
                alex.run()
                self.lbl_results.setText(alex.result)

            if self.lbl_work.text():
                self.lbl_results.setText("К сожалению данный функционал пока не поддерживается, попробуйте YOLO")
            if (self.lbl_work.text() is None or self.lbl_test.text()) is None or self.lbl_train.text() is None:
                self.lbl_results.setText('Укажите пути к файлам для обучения и тестирования')



def start():
    app = QApplication(sys.argv)
    ui = NN_UI()
    ui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    start()
