pre-commit run --all-files
pylama --ignore="E501,E722,C901,W291,E302,W391" --skip="venv/*,**/__init__.py,test/*"
