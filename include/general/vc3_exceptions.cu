#ifndef VC3_GENERAL_EXCEPTIONS
#define VC3_GENERAL_EXCEPTIONS

#include <string>

//Exceptions namespace
namespace vc3_exc
{
	//******** All errors
	class vc3_err
	{
		public:
		std::string name;
		const std::string getname() const {return name;};
	};

	//******** System errors
	class system_err: public vc3_err
	{
		public:
		system_err() {name="System error";}
	};

	class bad_alloc: public system_err
	{
		public:
		bad_alloc() {name="Error: memory allocation failed";}
	};

	class bad_file: public system_err
	{
		public:
		bad_file() {name="Error: file operation failed";}
	};

	//******** Logical errors
	class logical_err: public vc3_err
	{
		public:
		logical_err() {name="Logical error";}
	};

	//******** Logical errors: no_data errors
	class no_data: public logical_err
	{
		public:
		no_data() {name="No data error";}
	};

	class bad_pointer: public no_data
	{
		public:
		bad_pointer() {name="Error: NULL pointer is inadmissible";}
	}; //class bad_pointer

	class bad_index: public no_data
	{
		public:
		bad_index() {name=std::string("Error: Index is out of range");}
	}; //class bad_index

	//******** Logical errors: bad_data errors
	class bad_data: public logical_err
	{
		public:
		bad_data() {name=std::string("Error: Bad data");}
	};

	class bad_value: public bad_data
	{
		public:
		bad_value() {name=std::string("Error: invalid data value");}
		bad_value(std::string s) {name=s;}
	}; //class bad_value

	class data_missing: public bad_data
	{
		public:
		data_missing() {name=std::string("Error: Some data are missing");}
	}; //class data_missing

	class bad_size: public bad_data
	{
		public:
		bad_size() {name=std::string("Error: Invalid size");}
	}; //class bad_size

	//******** Logical errors: function_fail errors
	class function_fail: public logical_err
	{
		public:
		function_fail() {name=std::string("Error: Function fail");}
	};


} //namespace vc3_exc

#endif //VC3_GENERAL_EXCEPTIONS
