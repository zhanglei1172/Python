#!/usr/bin/env python
# -*- coding: utf-8 -*-
def demo_():
	# return 0
	print('1')
	yield 1
	yield 2
	print('1')

for i in demo_():
	print(i)
