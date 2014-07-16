#!/bin/bash

# Este script genera la documentacion
#
# Author: Exequiel Fuentes <efulet@gmail.com>
# Author: Brian Keith <briankeithn@gmail.com>

if which epydoc >/dev/null
then
  echo ''
else
  echo "Parece que epydoc no esta instalado en el sistema"
fi

BINPATH=`dirname $0`

epydoc --html ${BINPATH}/../pca/lib -v -o ${BINPATH}/../doc/apidocs --name PCA --graph all
