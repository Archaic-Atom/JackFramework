#!/bin/bash
echo $"Start to clean the project"
rm -r ./Source/build/
rm -r ./Source/dist/
rm -r ./Source/JackFramework.egg-info/
find . -iname "*.log" -exec rm -rf {} \;
find . -iname "*.pyc" -exec rm -rf {} \;
find . -iname "*__pycache__*" -exec rm -rf {} \;
echo $"Finish"
