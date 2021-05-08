// https://forums.developer.nvidia.com/t/writing-gl-textures-in-cuda-example-code-using-cudagraphicsglregisterimage-rather-than-pbo/15850
void test() {
    GLuint volumeTexture;

    int size = 16*16*16*4*4;

    GLubyte *data = new GLubyte;
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    glGenTextures(1, &volumeTexture);
    glBindTexture(GL_TEXTURE_3D, volumeTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    //Example 1: a float 32 rgba texture
    glTexImage3D(GL_TEXTURE_3D, 0,GL_RGBA32F, 16, 16,16,
                 0, GL_RGBA, GL_FLOAT,data);

    //Example 2: an int 32 rgba texture
    glTexImage3D(GL_TEXTURE_3D, 0,GL_RGBA32UI, 16, 16,16,
                 0,  GL_RGBA_INTEGER, GL_UNSIGNED_INT,data);

    //Example 3: an int 8 rgba texture
    glTexImage3D(GL_TEXTURE_3D, 0,GL_RGBA8UI, 16, 16,16,
                 0,  GL_RGBA_INTEGER, GL_UNSIGNED_BYTE,data);

    //Example 4: an int 8 red channel texture
    glTexImage3D(GL_TEXTURE_3D, 0,GL_R8UI, 16, 16,16,
                 0,  GL_RED_INTEGER, GL_UNSIGNED_BYTE,data);

    glBindTexture(GL_TEXTURE_3D, 0);


    struct cudaGraphicsResource* tex_CUDA;

    cudaError_t err =  cudaGraphicsGLRegisterImage(
            &tex_CUDA,
            volumeTexture,
            GL_TEXTURE_3D,
            cudaGraphicsMapFlagsNone);

    printf("cudaGraphicsGLRegisterImage error [%d]:",err);

    if ( err==cudaSuccess) printf( "cudaSuccess" );

    if ( err==cudaErrorInvalidDevice) printf( "cudaErrorInvalidDevice" );

    if ( err==cudaErrorInvalidValue) printf( "cudaErrorInvalidValue" );

    if ( err==cudaErrorInvalidResourceHandle) printf( "cudaErrorInvalidResourceHandle" );

    if ( err==cudaErrorUnknown) printf( "cudaErrorUnknown" );

    printf("\n");
}