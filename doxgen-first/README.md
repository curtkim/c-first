## Howto

    doxygen -g
    doxygen Doxyfile
    google-chrome doxygen/html/index.html

    # by docker
    docker run --rm -v $(pwd):/data hrektts/doxygen doxygen