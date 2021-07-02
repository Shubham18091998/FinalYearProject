#define F_CPU 1000000UL
#include <util/delay.h>
#include <avr/io.h>
#include <stdlib.h>
#include <string.h>
#include "ADC.h"
#include "lcd.h"
#define treshold 37
double thermister;
long r;
char buffer[20];
int main()
{
	int val;
	ADC_init();
	DDRC=0xFF;
	DDRB|=0x07;
	lcd_init();
	while (1)
	{
		val=ADC_read(0);
		r=((10230000/val)-10000);
		thermister=log(r);
		thermister= 1 / (0.001129148 + (0.000234125 * thermister) + (0.0000000876741 * thermister * thermister * thermister));
		thermister=thermister-273.15;
		memset(buffer,0,20);
		dtostrf(thermister,3,2,buffer);
		lcd_print(buffer);
		lcd_print("C");
		lcd_cmd(0xC0);
		if (thermister>treshold)
		{
			lcd_print("Alert");
			_delay_ms(500);
		}			
		lcd_init();
	}		
}