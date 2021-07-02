void ADC_init();
int ADC_read(char);
void ADC_init()
{
	DDRA=0x00;
	ADCSRA|=(1<<ADEN)|(1<<ADPS2)|(1<<ADPS1)|(1<<ADPS0);
	ADMUX=0x40;
}
int ADC_read(char ch)
{
	ADMUX=ADMUX|(ch & 0x0F);
	int ain,ainlow;
	ADCSRA|=(1<<ADSC);
	while ((ADCSRA&(1<<ADIF))==0);
	_delay_us(10);
	ainlow=(int)ADCL;
	ain=(int)ADCH*256;
	ain=ain+ainlow;
	return (ain);
}