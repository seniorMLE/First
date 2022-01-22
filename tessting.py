# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 03:29:44 2022

@author: Admin
"""
import os
from pathlib import Path

path = os.getcwd()
path = path+"\\agentBackup"
my_dir = Path(path)   
print(my_dir)

print(my_dir.is_file())