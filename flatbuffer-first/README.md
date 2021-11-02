# howto

    cd cmake-build-debug
    docker run -v $(pwd):/src -v $(pwd)/cmake-build-debug:/dest neomantra/flatbuffers:v2.0.0 flatc --cpp -o /dest /src/monster.fbs
    # generate monster_geneated.h

    docker run -v $(pwd):/src -v $(pwd)/cmake-build-debug:/dest neomantra/flatbuffers:v2.0.0 flatc --cpp -o /dest /src/person.fbs

    # deprecated
    ~/.conan/data/flatc/1.12.0/_/_/package/44fcf6b9a7fb86b2586303e3db40189d3b511830/bin/flatc --cpp ./monster.fbs
    
## reference
- https://copynull.tistory.com/411?category=892106

## mallochook관련 의문
- new operator override가 없으면 mallochook이 동작하지 않는다.
- flatbuffers::FlatBufferBuilder가 malloc이 아닌 다른 alloc을 하나?
