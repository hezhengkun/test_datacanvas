#encoding=utf-8

import os
import sys
import math

_mataclass_ = type


class Man():
	name = "hz"
	age = 29
	height = "169"

	#两个下划线开头的类成员变量，为私有变量，不能被类以外的函数访问。
	__salary = "20w"

	def run(self):
		print ("%s run very fast"%self.name)

	def money(self, __salary):
		print ("Her salary is %s"%__salary)



man1 = Man()
man1.run()

man1.name = "wxw"
man1.run()

man1.money("30w")


