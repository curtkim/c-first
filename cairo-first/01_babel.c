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
hex_color_t HI_COLOR_2 = { 0xd4, 0xd0, 0xc8 };
hex_color_t LO_COLOR_1 = { 0x80, 0x80, 0x80 };
hex_color_t LO_COLOR_2 = { 0x40, 0x40, 0x40 };
hex_color_t BLACK  = { 0, 0, 0 };

static void
set_hex_color (cairo_t *cr, hex_color_t color)
{
    cairo_set_source_rgb (cr,
                          color.r / 255.0,
                          color.g / 255.0,
                          color.b / 255.0);
}

static void
bevel_box (cairo_t *cr, int x, int y, int width, int height)
{
    cairo_save (cr);

    cairo_set_line_width (cr, 1.0);
    cairo_set_line_cap (cr, CAIRO_LINE_CAP_SQUARE);

    /* Fill and highlight */
    set_hex_color (cr, HI_COLOR_1);
    cairo_rectangle (cr, x, y, width, height);
    cairo_fill (cr);

    /* 2nd highlight */
    set_hex_color (cr, HI_COLOR_2);
    cairo_move_to (cr, x + 1.5, y + height - 1.5);
    cairo_rel_line_to (cr, width - 3, 0);
    cairo_rel_line_to (cr, 0, - (height - 3));
    cairo_stroke (cr);

    /* 1st lowlight */
    set_hex_color (cr, LO_COLOR_1);
    cairo_move_to (cr, x + .5, y + height - 1.5);
    cairo_rel_line_to (cr, 0, - (height - 2));
    cairo_rel_line_to (cr, width - 2, 0);
    cairo_stroke (cr);

    /* 2nd lowlight */
    set_hex_color (cr, LO_COLOR_2);
    cairo_move_to (cr, x + 1.5, y + height - 2.5);
    cairo_rel_line_to (cr, 0, - (height - 4));
    cairo_rel_line_to (cr, width - 4, 0);
    cairo_stroke (cr);

    cairo_restore (cr);
}

static void
bevel_circle (cairo_t *cr, int x, int y, int width)
{
    double radius = (width-1)/2.0 - 0.5;

    cairo_save (cr);

    cairo_set_line_width (cr, 1);

    /* Fill and Highlight */
    set_hex_color (cr, HI_COLOR_1);
    cairo_arc (cr, x+radius+1.5, y+radius+1.5, radius,
               0, 2* M_PI);
    cairo_fill (cr);

    /* 2nd highlight */
    set_hex_color (cr, HI_COLOR_2);
    cairo_arc (cr, x+radius+0.5, y+radius+0.5, radius,
               0, 2 * M_PI);
    cairo_stroke (cr);

    /* 1st lowlight */
    set_hex_color (cr, LO_COLOR_1);
    cairo_arc (cr, x+radius+0.5, y+radius+0.5, radius,
               3 * M_PI_4, 7 * M_PI_4);
    cairo_stroke (cr);

    /* 2nd lowlight */
    set_hex_color (cr, LO_COLOR_2);
    cairo_arc (cr, x+radius+1.5, y+radius+1.5, radius,
               3 * M_PI_4, 7 * M_PI_4);
    cairo_stroke (cr);

    cairo_restore (cr);
}

/* Slightly smaller than specified to match interior size of bevel_box */
static void
flat_box (cairo_t *cr, int x, int y, int width, int height)
{
    cairo_save (cr);

    /* Fill background */
    set_hex_color (cr, HI_COLOR_1);
    cairo_rectangle (cr, x+1, y+1, width-2, height-2);
    cairo_fill (cr);

    /* Stroke outline */
    cairo_set_line_width (cr, 1.0);
    set_hex_color (cr, BLACK);
    cairo_rectangle (cr, x + 1.5, y + 1.5, width - 3, height - 3);
    cairo_stroke (cr);

    cairo_restore (cr);
}

static void
flat_circle (cairo_t *cr, int x, int y, int width)
{
    double radius = (width - 1) / 2.0;

    cairo_save (cr);

    /* Fill background */
    set_hex_color (cr, HI_COLOR_1);
    cairo_arc (cr, x+radius+0.5, y+radius+0.5, radius-1,
               0, 2 * M_PI);
    cairo_fill (cr);

    /* Fill background */
    cairo_set_line_width (cr, 1.0);
    set_hex_color (cr, BLACK);
    cairo_arc (cr, x+radius+0.5, y+radius+0.5, radius-1,
               0, 2 * M_PI);
    cairo_stroke (cr);

    cairo_restore (cr);
}

static void
groovy_box (cairo_t *cr, int x, int y, int width, int height)
{
    cairo_save (cr);

    /* Highlight */
    set_hex_color (cr, HI_COLOR_1);
    cairo_set_line_width (cr, 2);
    cairo_rectangle (cr, x + 1, y + 1, width - 2, height - 2);
    cairo_stroke (cr);

    /* Lowlight */
    set_hex_color (cr, LO_COLOR_1);
    cairo_set_line_width (cr, 1);
    cairo_rectangle (cr, x + 0.5, y + 0.5, width - 2, height - 2);
    cairo_stroke (cr);

    cairo_restore (cr);
}

#define CHECK_BOX_SIZE 13
#define CHECK_COLOR BLACK

typedef enum {UNCHECKED, CHECKED} checked_status_t;

static void
check_box (cairo_t *cr, int x, int y, checked_status_t checked)
{
    cairo_save (cr);

    bevel_box (cr, x, y, CHECK_BOX_SIZE, CHECK_BOX_SIZE);

    if (checked) {
        set_hex_color (cr, CHECK_COLOR);
        cairo_move_to (cr, x + 3, y + 5);
        cairo_rel_line_to (cr, 2.5, 2);
        cairo_rel_line_to (cr, 4.5, -4);
        cairo_rel_line_to (cr, 0, 3);
        cairo_rel_line_to (cr, -4.5, 4);
        cairo_rel_line_to (cr, -2.5, -2);
        cairo_close_path (cr);
        cairo_fill (cr);
    }

    cairo_restore (cr);
}

#define RADIO_SIZE CHECK_BOX_SIZE
#define RADIO_DOT_COLOR BLACK
static void
radio_button (cairo_t *cr, int x, int y, checked_status_t checked)
{

    cairo_save (cr);

    bevel_circle (cr, x, y, RADIO_SIZE);

    if (checked) {
        set_hex_color (cr, RADIO_DOT_COLOR);
        cairo_arc (cr,
                   x + (RADIO_SIZE-1) / 2.0 + 0.5,
                   y + (RADIO_SIZE-1) / 2.0 + 0.5,
                   (RADIO_SIZE-1) / 2.0 - 3.5,
                   0, 2 * M_PI);
        cairo_fill (cr);
    }

    cairo_restore (cr);
}

static void
draw_bevels (cairo_t *cr, int width, int height)
{
    int check_room = (width - 20) / 3;
    int check_pad = (check_room - CHECK_BOX_SIZE) / 2;

    groovy_box (cr, 5, 5, width - 10, height - 10);

    check_box (cr, 10+check_pad, 10+check_pad, UNCHECKED);
    check_box (cr, check_room+10+check_pad, 10+check_pad, CHECKED);
    flat_box (cr, 2 * check_room+10+check_pad, 10+check_pad,
              CHECK_BOX_SIZE, CHECK_BOX_SIZE);

    radio_button (cr, 10+check_pad, check_room+10+check_pad, UNCHECKED);
    radio_button (cr, check_room+10+check_pad, check_room+10+check_pad, CHECKED);
    flat_circle (cr, 2 * check_room+10+check_pad, check_room+10+check_pad, CHECK_BOX_SIZE);
}

int
main (void)
{
    cairo_t *cr;
    cairo_surface_t *surface;

    surface = cairo_image_surface_create_for_data (image, CAIRO_FORMAT_ARGB32,
                                                   WIDTH, HEIGHT, STRIDE);
    cr = cairo_create (surface);

    cairo_rectangle (cr, 0, 0, WIDTH, HEIGHT);
    set_hex_color (cr, BG_COLOR);
    cairo_fill (cr);

    draw_bevels (cr, WIDTH, HEIGHT);

    cairo_surface_write_to_png (surface, "bevels.png");

    cairo_destroy (cr);

    cairo_surface_destroy (surface);

    return 0;
}

