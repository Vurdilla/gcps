#ifndef V3_GENERAL_STRINGEXT
#define V3_GENERAL_STRINGEXT

#include <string>
#include <algorithm>
#include <sstream>

#include "../types.cu"

namespace vc3_general {

	const size_t _itos_size=10;
	char* itoa(int x, char* result, int base=10);
	std::string itoa(int x, int base=10);
	char* lltoa(long long int x, char* result, int base=10);
	std::string lltoa(long long int x, int base=10);

	const size_t defsignsafterpoint_flt=6;

	std::string ftos(flt, size_t signsafterpoint = defsignsafterpoint_flt) throw();
	std::string ftos_sp(flt, size_t signsafterpoint = defsignsafterpoint_flt) throw();
	std::string ftos(flt2, size_t signsafterpoint = defsignsafterpoint_flt) throw();
	std::string ftos_sp(flt2, size_t signsafterpoint = defsignsafterpoint_flt) throw();
	int stoi(std::string) throw();
	flt stof(std::string) throw();
	__host__ int stodv(std::string s, std::vector<flt>* v, char sep = ',') throw();
	__host__ int stodv(std::string s, std::vector<flt2>* v, char sep = ',') throw();

} //namespace vc3_general

/**
* C++ version 0.4 char* style "itoa":
* Written by Lukás Chmela
* Released under GPLv3.
*/
char* vc3_general::itoa(int value, char* result, int base) {
    // check that the base if valid
	if (base < 2 || base > 36) { *result = '\0'; return result; }
	char* ptr = result, *ptr1 = result, tmp_char;
	int tmp_value;
	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
	} while ( value );

    // Apply negative sign
	if (tmp_value < 0) *ptr++ = '-';
	*ptr-- = '\0';
	while(ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--= *ptr1;
		*ptr1++ = tmp_char;
	}
	return result;
}

/**
	 * C++ version 0.4 std::string style "itoa":
	 * Contributions from Stuart Lowe, Ray-Yuan Sheu,
	 * Rodrigo de Salvo Braz, Luc Gallant, John Maloney
	 * and Brian Hunt
	 */
std::string vc3_general::itoa(int value, int base) {
    std::string buf;

	// check that the base if valid
	if (base < 2 || base > 16) return buf;

	enum { kMaxDigits = 35 };
	buf.reserve( kMaxDigits ); // Pre-allocate enough space.

	int quotient = value;

	// Translating number to string with base:
	do {
		buf += "0123456789abcdef"[ std::abs( quotient % base ) ];
		quotient /= base;
	} while ( quotient );

    // Append the negative sign
	if ( value < 0) buf += '-';

	std::reverse( buf.begin(), buf.end() );
	return buf;
}

char* vc3_general::lltoa(long long int value, char* result, int base) {
    // check that the base if valid
	if (base < 2 || base > 36) { *result = '\0'; return result; }
	char* ptr = result, *ptr1 = result, tmp_char;
	long long int tmp_value;
	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
	} while ( value );

    // Apply negative sign
	if (tmp_value < 0) *ptr++ = '-';
	*ptr-- = '\0';
	while(ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--= *ptr1;
		*ptr1++ = tmp_char;
	}
	return result;
}

std::string vc3_general::lltoa(long long int value, int base) {
    std::string buf;

	// check that the base if valid
	if (base < 2 || base > 16) return buf;

	enum { kMaxDigits = 35 };
	buf.reserve( kMaxDigits ); // Pre-allocate enough space.

	long long int quotient = value;

	// Translating number to string with base:
	do {
		buf += "0123456789abcdef"[ std::abs( quotient % base ) ];
		quotient /= base;
	} while ( quotient );

    // Append the negative sign
	if ( value < 0) buf += '-';

	std::reverse( buf.begin(), buf.end() );
	return buf;
}

std::string vc3_general::ftos(flt x, size_t sap) throw()
{
	std::string s((x>=0)?"":"-");
	int t=x<0?-x:x;
	s+=itoa(t);
	if(sap>0) s+=".";
	else return s;

	x=(x<0?-x:x)-t;
    for(size_t q=0;q<sap;q++)
	{
		x*=10;
		t=x;
		s+=itoa(t);
		x-=flt(t);
	}
	return s;
}

std::string vc3_general::ftos_sp(flt x, size_t sap) throw()
{
	std::string s((x>=0)?" ":"-");
	int t=x<0?-x:x;
	s+=itoa(t);
	if(sap>0) s+=".";
	else return s;

	x=(x<0?-x:x)-t;
    for(size_t q=0;q<sap;q++)
	{
		x*=10;
		t=x;
		s+=itoa(t);
		x-=flt(t);
	}
	return s;
}

std::string vc3_general::ftos(flt2 x, size_t sap) throw()
{
	std::string s((x >= 0) ? "" : "-");
	int t = x < 0 ? -x : x;
	s += itoa(t);
	if (sap > 0) s += ".";
	else return s;

	x = (x < 0 ? -x : x) - t;
	for (size_t q = 0; q < sap; q++)
	{
		x *= 10;
		t = x;
		s += itoa(t);
		x -= flt2(t);
	}
	return s;
}

std::string vc3_general::ftos_sp(flt2 x, size_t sap) throw()
{
	std::string s((x >= 0) ? " " : "-");
	int t = x < 0 ? -x : x;
	s += itoa(t);
	if (sap > 0) s += ".";
	else return s;

	x = (x < 0 ? -x : x) - t;
	for (size_t q = 0; q < sap; q++)
	{
		x *= 10;
		t = x;
		s += itoa(t);
		x -= flt2(t);
	}
	return s;
}

int vc3_general::stoi(std::string s) throw()
{
        const char* c=s.c_str();
        return std::atoi(c);
}

flt vc3_general::stof(std::string s) throw()
{
        const char* c=s.c_str();
        return std::atof(c);
}

__host__ int vc3_general::stodv(std::string s, std::vector<flt>* v, char sep) throw()
{
	std::stringstream ss(s);
	std::string token;
	v->clear();
	while (std::getline(ss, token, sep)) {
		// Convert the token to an integer and store it in the vector
		flt value = vc3_general::stof(token);
		v->push_back(value);
	}
	return v->size();
}

__host__ int vc3_general::stodv(std::string s, std::vector<flt2>* v, char sep) throw()
{
	std::stringstream ss(s);
	std::string token;
	v->clear();
	while (std::getline(ss, token, sep)) {
		// Convert the token to an integer and store it in the vector
		flt2 value = vc3_general::stof(token);
		v->push_back(value);
	}
	return v->size();
}

#endif //#ifndef V3_GENERAL_STRINGEXT

