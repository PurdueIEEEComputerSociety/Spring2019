from PyQt5 import QtCore, QtGui, QtWidgets
import sys

from ML_Paint import Ui_StyleTransferGUI

from NetworkTF import *

import matplotlib.pyplot as plt

class StyleTransferGUI(Ui_StyleTransferGUI):
	def __init__(self, w):
		Ui_StyleTransferGUI.__init__(self)
		Ui_StyleTransferGUI.setupUi(self, w)
		self.window = w

		self.model = StyleTransferModel(numIterations=20)

		self.btn_start.clicked.connect(self.runStyle)
		self.btn_load.clicked.connect(self.loadFiles)

	def loadFiles(self):
		# self.contentFileName = 'scaledContent/29.jpg'
		# self.styleFileName = 'scaledStyle/17.jpg'
		# self.outputFileName = 'styled.png'
		self.contentFileName = self.edt_content.text()
		self.styleFileName = self.edt_style.text()
		self.outputFileName = self.edt_output.text()
		
		#TODO: Input validation
		self.model.setStyleImage(self.styleFileName)
		self.model.setContentImage(self.contentFileName)
		
		styleScene = QtWidgets.QGraphicsScene()            
		styleScene.addPixmap(QtGui.QPixmap(QtGui.QImage(self.styleFileName)))
		self.pic_style.setScene(styleScene)
		self.pic_style.fitInView(styleScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

		contentScene = QtWidgets.QGraphicsScene()            
		contentScene.addPixmap(QtGui.QPixmap(QtGui.QImage(self.contentFileName)))
		self.pic_content.setScene(contentScene)
		self.pic_content.fitInView(contentScene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

	def runStyle(self):
		styled = self.model.transfer()
		styled = styled.astype('uint8')
		fig = plt.figure(frameon=False)
		ax = plt.Axes(fig, [0.,0.,1.,1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		plt.imshow(styled, aspect='normal')
		plt.savefig(self.outputFileName)

		outputScene = QtWidgets.QGraphicsScene()            
		outputScene.addPixmap(QtGui.QPixmap(QtGui.QImage(self.outputFileName)))
		self.pic_output.setScene(outputScene)
		self.pic_output.fitInView(outputScene.itemsBoundingRect())



if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	w = QtWidgets.QMainWindow()
	window = StyleTransferGUI(w)
	w.show()
	sys.exit(app.exec_())