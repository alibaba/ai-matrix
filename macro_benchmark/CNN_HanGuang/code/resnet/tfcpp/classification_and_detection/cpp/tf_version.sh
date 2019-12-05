pip list 2>>/dev/null | grep tensorflow | sed "s/tensorflow//" | tr -d ' '
