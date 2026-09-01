// kernel.c (freestanding, x86 32-bit)
#include <stdint.h>
#include <stddef.h>

/* ========= VGA TEXT OUTPUT (80x25) ========= */
static volatile uint16_t* const VGA = (uint16_t*)0xB8000;
static uint8_t vga_row = 0, vga_col = 0;
static uint8_t vga_color = 0x0F; // white on black

static inline uint16_t vga_entry(char c, uint8_t color) {
    return (uint16_t)c | ((uint16_t)color << 8);
}

static void vga_clear(void) {
    for (size_t y = 0; y < 25; y++) {
        for (size_t x = 0; x < 80; x++) {
            VGA[y * 80 + x] = vga_entry(' ', vga_color);
        }
    }
    vga_row = 0; vga_col = 0;
}

static void vga_putc(char c) {
    if (c == '\n') {
        vga_col = 0;
        if (vga_row < 24) vga_row++;
        return;
    }
    VGA[vga_row * 80 + vga_col] = vga_entry(c, vga_color);
    vga_col++;
    if (vga_col >= 80) {
        vga_col = 0;
        if (vga_row < 24) vga_row++;
    }
}

static void vga_print(const char* s) {
    while (*s) vga_putc(*s++);
}

static void vga_print_hex32(uint32_t v) {
    const char* hex = "0123456789ABCDEF";
    vga_print("0x");
    for (int i = 7; i >= 0; --i) {
        uint8_t nyb = (v >> (i * 4)) & 0xF;
        vga_putc(hex[nyb]);
    }
}

/* ========= YOUR 100-MEMORY WORLD =========
   mem[0..80] : 9x9 grid (81)
   mem[81..99]: meta / gbmc / io / logs etc (19)
*/
typedef struct {
    int8_t  state;     // e.g., -128..127
    int8_t  energy;    // -128..127
    int8_t  tag;       // arbitrary label
    uint8_t flags;     // bitflags
    int16_t v1;        // e.g., responsibility vector x
    int16_t v2;        // e.g., responsibility vector y
    uint8_t next;      // 0..99
    uint8_t ttl;       // 0..255
} Cell;

static Cell mem[100];

static inline uint8_t grid_index(uint8_t x, uint8_t y) { // x,y in 0..8
    return (uint8_t)(y * 9 + x); // 0..80
}

static void mem_init(void) {
    // zero everything (freestanding: do manually)
    for (int i = 0; i < 100; i++) {
        mem[i].state = 0;
        mem[i].energy = 0;
        mem[i].tag = 0;
        mem[i].flags = 0;
        mem[i].v1 = 0;
        mem[i].v2 = 0;
        mem[i].next = (uint8_t)i;
        mem[i].ttl = 0;
    }

    // init 9x9 grid with a simple pattern for sanity check
    for (uint8_t y = 0; y < 9; y++) {
        for (uint8_t x = 0; x < 9; x++) {
            uint8_t idx = grid_index(x, y);
            mem[idx].state  = (int8_t)((x + y) & 1); // checker
            mem[idx].energy = 1;
            mem[idx].tag    = 42; // arbitrary
            mem[idx].ttl    = 10;
        }
    }

    // meta slots example
    mem[99].state = 7;    // mode / version
    mem[99].energy = 1;   // alive flag
    mem[99].v1 = 100;     // "max memory"
}

static void draw_grid_9x9(void) {
    vga_print("Grid(9x9) state:\n");
    for (uint8_t y = 0; y < 9; y++) {
        for (uint8_t x = 0; x < 9; x++) {
            uint8_t idx = grid_index(x, y);
            char c = (mem[idx].state != 0) ? '#' : '.';
            vga_putc(c);
            vga_putc(' ');
        }
        vga_putc('\n');
    }
}

void kmain(void) {
    vga_clear();
    vga_print("GBMC-MinKernel booted.\n");
    vga_print("mem[100] init...\n");

    mem_init();

    vga_print("mem[99].v1 (max) = ");
    vga_print_hex32((uint32_t)(uint16_t)mem[99].v1);
    vga_putc('\n');

    draw_grid_9x9();

    vga_print("\nOK. (halt)\n");
    for (;;) {
        __asm__ volatile ("hlt");
    }
}
