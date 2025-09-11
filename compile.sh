#!/bin/bash

# Проверяем, передан ли аргумент
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Получаем имя файла из аргумента
filename=$1

# Если имя файла не имеет расширения .tex, добавим его
if [[ "$filename" != *.tex ]]; then
    filename="$filename.tex"
fi

# Проверяем, существует ли файл
if [ ! -f "$filename" ]; then
    echo "File $filename not found!"
    exit 1
fi

# Извлекаем базовое имя без расширения для использования в bibtex
basename=$(basename "$filename" .tex)


# Первая компиляция
echo "First pdflatex run..."
pdflatex -synctex=1 -interaction=nonstopmode -aux-directory=./aux_dir -output-directory=./pdfs "$filename" 

# Обработка библиографии
echo "Running bibtex..."
bibtex "./aux_dir/$basename.aux"

# Вторая компиляция
echo "Second pdflatex run..."
pdflatex -synctex=1 -interaction=nonstopmode -aux-directory=./aux_dir -output-directory=./pdfs "$filename"

# Третья компиляция
echo "Third pdflatex run..."
pdflatex -synctex=1 -interaction=nonstopmode -aux-directory=./aux_dir -output-directory=./pdfs "$filename"

echo "Compilation complete. The output is in the ./pdfs directory."