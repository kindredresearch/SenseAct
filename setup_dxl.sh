#!/bin/bash

# Get script directory to help with
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export DXL_DIR="$DIR/senseact/lib/"

# Download DynamixelSDK 3.5.4. This is an older version. The newer versions do not use ctypes based python drivers.
# We found C-based serial drivers crucial for repeatable experiments with DXL. We could never achieve the same results with pyserial.
wget https://github.com/ROBOTIS-GIT/DynamixelSDK/archive/3.5.4.tar.gz -P $DXL_DIR

cd $DXL_DIR
mkdir DynamixelSDK && tar xf 3.5.4.tar.gz -C DynamixelSDK --strip-components 1
echo "DynamixelSDK repo extracted to $PWD"

# Delete tar file after extraction
rm 3.5.4.tar.gz

# Define C code path
DXL_BASE_PATH="$PWD/DynamixelSDK/c/build"
MACHINE_TYPE=`uname -m`
flag=true

# Find OS type to compile appropriate C code
if [[ "$OSTYPE" == "linux-gnu" ]]; then
        if [ ${MACHINE_TYPE} == 'x86_64' ]; then
		DXL_COMPILE_PATH="$DXL_BASE_PATH/linux64"
	else
		DXL_COMPILE_PATH="$DXL_BASE_PATH/linux32"
	fi
	SHARED_OBJ_TYPE="*.so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        DXL_COMPILE_PATH="$DXL_BASE_PATH/mac"
	SHARED_OBJ_TYPE="*.dylib"
elif [[ "$OSTYPE" == "cygwin" ]]; then
        echo "Unfortunately, the driver doesn't support cygwin. Feel free to check the code and modify it for your specs."
	exit 1
elif [[ "$OSTYPE" == "msys" ]]; then
        DXL_COMPILE_PATH="$DXL_BASE_PATH/win64"
	SHARED_OBJ_TYPE="*.dll"
	flag=false
elif [[ "$OSTYPE" == "win32" ]]; then
        DXL_COMPILE_PATH="$DXL_BASE_PATH/win32"
	SHARED_OBJ_TYPE="*.dll"
	flag=false
elif [[ "$OSTYPE" == "freebsd"* ]]; then
        echo "Unfortunately, the driver doesn't support freebsd. Feel free to check the code and modify it for your specs."
	exit 1
else
        echo "I have no idea!"
	exit 1
fi

# Compile C libraries
cd $DXL_COMPILE_PATH
if [ "$flag" = true ]; then
	make
	sudo make install
fi

SHARED_OBJ_PATH="$(find $DXL_BASE_PATH -iname $SHARED_OBJ_TYPE)"

DXL_PYTHON_PATH="$DXL_BASE_PATH/../../python/dynamixel_functions_py"
cd $DXL_PYTHON_PATH
PY_FILE="dynamixel_functions.py"

# Delete lines
sed -i '24,29d' $PY_FILE

# Insert line
sed -i "24i #Shared object path" $PY_FILE
sed -i "25idxl_lib = cdll.LoadLibrary(\"$SHARED_OBJ_PATH\")" $PY_FILE

echo "DynamixelSDK setup successful!"





