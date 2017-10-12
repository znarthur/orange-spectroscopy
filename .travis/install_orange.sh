if [ $ORANGE == "release" ]; then
    echo "Orange: Skipping separate Orange install"
    return 0
fi

if [ $ORANGE == "master" ]; then
    echo "Orange: from git master"
    pip install git+git://github.com/biolab/orange3.git
    return $?;
fi

PACKAGE="orange3==$ORANGE"
echo "Orange: installing $PACKAGE"
pip install $PACKAGE