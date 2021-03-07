from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import pickle

model = pickle.load(open('cart_model.pkl', 'rb'))


class Ui_Form(QtWidgets.QWidget):

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1093, 779)
        self.lineEdit_10 = QtWidgets.QLineEdit(Form)
        self.lineEdit_10.setGeometry(QtCore.QRect(310, 70, 131, 31))
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(240, 80, 55, 16))
        self.label_2.setObjectName("label_2")
        self.lineEdit_16 = QtWidgets.QLineEdit(Form)
        self.lineEdit_16.setGeometry(QtCore.QRect(310, 120, 131, 31))
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(240, 130, 55, 16))
        self.label_3.setObjectName("label_3")
        self.lineEdit_17 = QtWidgets.QLineEdit(Form)
        self.lineEdit_17.setGeometry(QtCore.QRect(310, 170, 131, 31))
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(240, 180, 55, 16))
        self.label_4.setObjectName("label_4")
        self.lineEdit_18 = QtWidgets.QLineEdit(Form)
        self.lineEdit_18.setGeometry(QtCore.QRect(310, 220, 131, 31))
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(240, 230, 55, 16))
        self.label_5.setObjectName("label_5")
        self.lineEdit_19 = QtWidgets.QLineEdit(Form)
        self.lineEdit_19.setGeometry(QtCore.QRect(310, 280, 131, 31))
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(240, 290, 55, 16))
        self.label_6.setObjectName("label_6")
        self.lineEdit_20 = QtWidgets.QLineEdit(Form)
        self.lineEdit_20.setGeometry(QtCore.QRect(310, 340, 131, 31))
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(240, 350, 55, 16))
        self.label_7.setObjectName("label_7")
        self.lineEdit_11 = QtWidgets.QLineEdit(Form)
        self.lineEdit_11.setGeometry(QtCore.QRect(670, 70, 131, 31))
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(590, 80, 75, 16))
        self.label_8.setObjectName("label_8")
        self.lineEdit_21 = QtWidgets.QLineEdit(Form)
        self.lineEdit_21.setGeometry(QtCore.QRect(670, 120, 131, 31))
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(580, 130, 75, 16))
        self.label_9.setObjectName("label_9")
        self.lineEdit_22 = QtWidgets.QLineEdit(Form)
        self.lineEdit_22.setGeometry(QtCore.QRect(670, 170, 131, 31))
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.lineEdit_23 = QtWidgets.QLineEdit(Form)
        self.lineEdit_23.setGeometry(QtCore.QRect(670, 220, 131, 31))
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.lineEdit_24 = QtWidgets.QLineEdit(Form)
        self.lineEdit_24.setGeometry(QtCore.QRect(670, 280, 131, 31))
        self.lineEdit_24.setObjectName("lineEdit_24")
        self.lineEdit_25 = QtWidgets.QLineEdit(Form)
        self.lineEdit_25.setGeometry(QtCore.QRect(670, 330, 131, 31))
        self.lineEdit_25.setObjectName("lineEdit_25")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(570, 180, 75, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(570, 230, 75, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setGeometry(QtCore.QRect(570, 290, 75, 16))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setGeometry(QtCore.QRect(570, 340, 75, 16))
        self.label_13.setObjectName("label_13")
        self.lineEdit_26 = QtWidgets.QLineEdit(Form)
        self.lineEdit_26.setGeometry(QtCore.QRect(670, 380, 131, 31))
        self.lineEdit_26.setObjectName("lineEdit_26")
        self.label_14 = QtWidgets.QLabel(Form)
        self.label_14.setGeometry(QtCore.QRect(570, 390, 70, 16))
        self.label_14.setObjectName("label_14")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(240, 460, 571, 61))
        self.pushButton.setObjectName("pushButton")
        self.label_15 = QtWidgets.QLabel(Form)
        self.label_15.setGeometry(350, 550, 400, 200)
        self.label_15.setFont(QtGui.QFont("Times", 10))

        self.setWindowTitle("Diabet Predict")
        self.pushButton.clicked.connect(self.click)
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
    def click(self):
        self.sonuc = [int(self.lineEdit_10.text()),
                      int(self.lineEdit_16.text()),
                      int(self.lineEdit_17.text()),
                      int(self.lineEdit_18.text()),
                      int(self.lineEdit_19.text()),
                      int(self.lineEdit_20.text()),
                      int(self.lineEdit_11.text()),
                      int(self.lineEdit_21.text()),
                      int(self.lineEdit_22.text()),
                      int(self.lineEdit_23.text()),
                      int(self.lineEdit_24.text()),
                      int(self.lineEdit_25.text()),
                      int(self.lineEdit_26.text())
                      ]

        self.pred = model.predict(pd.DataFrame(self.sonuc).T)[0]
        if self.pred == 0:
            self.label_15.setText("Diyabet hastası değildir.\nModelin anlamlı bulduğu degiskene göre tavsiye verilir."
                                  "\nŞeker düşmanı olmalısınız.\nBolca hareket etmelisiniz."
                                  "\nMeyve'yi dengeli ne az ne çok yemelisiniz."
                                  "\nAra öğünleri atlamamalısınız vs...")
        else:
            self.label_15.setText("Diyabet hastasidir.\nİnsulin, Glukoz ya da ikisi birden normal değerlere sahip olmayabilir."
                                  "\nModelin tahmin etmek için anlamlı bulduğu değişkenlere göre "
                                  "\nyorum yapılmıştır.")

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Diabet Predict"))
        self.label_2.setText(_translate("Form", "Glucose"))
        self.label_3.setText(_translate("Form", "BloodPres"))
        self.label_4.setText(_translate("Form", "SkinThicknes"))
        self.label_5.setText(_translate("Form", "Insulin"))
        self.label_6.setText(_translate("Form", "BMI"))
        self.label_7.setText(_translate("Form", "DiaPedFunc"))
        self.label_8.setText(_translate("Form", "Age"))
        self.label_9.setText(_translate("Form", "Preg_Age"))
        self.label_10.setText(_translate("Form", "Glucose_BMI"))
        self.label_11.setText(_translate("Form", "Insu_Glucose"))
        self.label_12.setText(_translate("Form", "Insu_BMI"))
        self.label_13.setText(_translate("Form", "Insu_Age"))
        self.label_14.setText(_translate("Form", "NewInsScor"))
        self.pushButton.setText(_translate("Form", "Tahmin Et"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()

    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())






