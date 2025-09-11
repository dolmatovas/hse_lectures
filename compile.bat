@echo off
setlocal

if "%~1"=="" (
    echo Ошибка: Укажите имя .tex файла
    echo Использование: %0 ^<имя_файла^>
    exit /b 1
)

set filename=%~1
set basename=%~n1

echo Compiling... %filename%...

pdflatex -synctex=1 -interaction=nonstopmode -aux-directory=./aux_dir -output-directory=./pdfs "%filename%"

if exist "output\%basename%.aux" (
    echo Обработка библиографии...
    bibtex "output\%basename%.aux"
)

pdflatex -synctex=1 -interaction=nonstopmode -aux-directory=./aux_dir -output-directory=./pdfs "%filename%"
pdflatex -synctex=1 -interaction=nonstopmode -aux-directory=./aux_dir -output-directory=./pdfs "%filename%"

echo "Ready!"
endlocal