#!/usr/bin/python3

import sys
from os import listdir, getcwd
from os.path import isfile, join
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QGroupBox, QCheckBox, QComboBox)
from PyQt5.QtCore import pyqtSlot


class MainWindow(QWidget):
  
  def __init__(self):
    super().__init__()
    
    self.init_variables()
    self.initUI()
    
  def init_variables(self):
    self.dir_path = getcwd()
    self.file_names = [f for f in listdir(self.dir_path) if isfile(join(self.dir_path, f))]
  
  def initUI(self):
    
    vbox = QVBoxLayout()
    hbox = QHBoxLayout()
    
    dir_button = QPushButton('Select Working Directory', self)
    dir_button.clicked.connect(self.find_dir)
    
    self.dir_display = QLabel(self.dir_path, self)
    
    hbox.addWidget(dir_button)
    hbox.addWidget(self.dir_display)
    
    self.file_list = QGroupBox("Files in Directory")
    self.show_filenames()
    self.file_list.setLayout(self.file_list_input)
    
    
    vbox.addLayout(hbox)
    vbox.addWidget(self.file_list)
    
    self.setLayout(vbox)
    
    self.setWindowTitle('RefNX SLD Evaluator')
    self.show()

  @pyqtSlot()
  def find_dir(self):
    self.dir_path = str(QFileDialog.getExistingDirectory(self,'Select Working Directory', self.dir_path))
    self.dir_display.setText(self.dir_path)
    self.show_filenames()
  
class FileDisplay(QGroupBox):
  
  def __init__(self, files):
    super().__init__('Filer')
    
    self.init_file_display(files)
    
  def init_file_display(self, files):
    self.layout = QVBoxLayout()
    self.fill_file_display(files)
    self.setLayout(self.layout)
    
  def fill_file_display(self, files):
    for fn in files:
      self.layout.addLayout(FileLine(fn))
  
class FileLine(QHBoxLayout):
  
  def __init__(self, fn):
    super().__init__()
    
    self.init_variables()
    self.init_file_line(fn)
  
  def init_variables(self):
    self.file_types = ['refnx',]
  
  def init_file_line(self, fn):
    self.active_button = QCheckBox(fn)
    self.load_type = QComboBox()
    
    for i in self.file_types:
      self.load_type.addItem(i)
    
    self.addWidget(self.active_button)
    self.addWidget(self.load_type)

if __name__ == '__main__':
  
  app = QApplication(sys.argv)
  app.setStyle('Fusion')
  
  main_window = MainWindow()
  
  sys.exit(app.exec_())
