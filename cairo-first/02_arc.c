#include <cairo.h>
#include <math.h>
#include <stdint.h>

#define WIDTH 100
#define HEIGHT 70
#define STRIDE (WIDTH * 4)

unsigned char image[STRIDE*HEIGHT];

typedef struct hex_color
{
    uint16_t r, g, b;
} hex_color_t;

hex_color_t BG_COLOR =  { 0xd4, 0xd0, 0xc8 };
hex_color_t HI_COLOR_1 = { 0xff, 0xff, 0xff };

hex_color_t BLACK  = { 0, 0, 0 };

static void
set_hex_color (cairo_t *cr, hex_color_t color)
{
    cairo_set_source_rgb (cr,
                          color.r / 255.0,
                          color.g / 255.0,
                          color.b / 255.0);
}

void draw(cairo_t *cr, int width, int height) {
    set_hex_color (cr, HI_COLOR_1);

    double xc = 128.0;
    double yc = 128.0;
    double radius = 100.0;
    double angle1 = 45.0  * (M_PI/180.0);  /* angles are specified */
    double angle2 = 180.0 * (M_PI/180.0);  /* in radians           */

    cairo_set_line_width (cr, 10.0);
    cairo_arc (cr, xc, yc, radius, angle1, angle2);
    cairo_stroke (cr);

    /* draw helping lines */
    cairo_set_source_rgba (cr, 1, 0.2, 0.2, 0.6);
    cairo_set_line_width (cr, 6.0);

    cairo_arc (cr, xc, yc, 10.0, 0, 2*M_PI);
    cairo_fill (cr);

    cairo_arc (cr, xc, yc, radius, angle1, angle1);
    cairo_line_to (cr, xc, yc);
    cairo_arc (cr, xc, yc, radius, angle2, angle2);
    cairo_line_to (cr, xc, yc);
    cairo_stroke (cr);

}

int main (void) {
    cairo_t *cr;
    cairo_surface_t *surface;
    surface = cairo_image_surface_create_for_data (image, CAIRO_FORMAT_ARGB32,
                                                   WIDTH, HEIGHT, STRIDE);
    cr = cairo_create (surface);
    cairo_rectangle (cr, 0, 0, WIDTH, HEIGHT);
    set_hex_color (cr, BG_COLOR);
    cairo_fill (cr);


    draw(cr, WIDTH, HEIGHT);

    cairo_surface_write_to_png (surface, "arc.png");

    cairo_destroy (cr);

    cairo_surface_destroy (surface);

    return 0;
}

