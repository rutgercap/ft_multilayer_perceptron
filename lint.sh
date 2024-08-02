pre-commit run --all-files
pylama --ignore="E501,E722,C901,W291" --skip="venv/*,**/__init__.py"
