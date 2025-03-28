// Copyright Axelera AI, 2025
#pragma once

#include <opencv2/opencv.hpp>
#include "AxDataInterface.h"

#include <optional>

using namespace std::string_literals;

namespace Ax
{

inline std::string
get_typename(unsigned char)
{
  return "uint8";
}

inline std::string
get_typename(signed char)
{
  return "int8";
}

inline std::string
get_typename(unsigned short)
{
  return "uint16";
}

inline std::string
get_typename(signed short)
{
  return "int16";
}

inline std::string
get_typename(unsigned int)
{
  return "unsigned int";
}

inline std::string
get_typename(signed int)
{
  return "int";
}

inline std::string
get_typename(unsigned long int)
{
  return "unsigned long";
}

inline std::string
get_typename(signed long int)
{
  return "long";
}

inline std::string
get_typename(float)
{
  return "float";
}

inline std::string
get_typename(double)
{
  return "double";
}

inline std::string
get_typename(std::string)
{
  return "string";
}

inline int
opencv_type_u8(AxVideoFormat format)
{
  return CV_MAKETYPE(CV_8U, AxVideoFormatNumChannels(format));
}

inline int
opencv_type_f32(AxVideoFormat format)
{
  return CV_MAKETYPE(CV_32F, AxVideoFormatNumChannels(format));
}

using properties = const std::unordered_map<std::string, std::string>;

template <typename T>
bool
convert_value(const std::string &s, T &value)
{
  //  TODO, trim the string and ensure all consumed
  return static_cast<bool>(std::istringstream(s) >> value);
}

template <typename LargeT, typename T>
bool
range_checked_convert_value(const std::string &s, T &value)
{
  LargeT i;
  if (!convert_value(s, i) || i < LargeT{ std::numeric_limits<T>::min() }
      || i > LargeT{ std::numeric_limits<T>::max() }) {
    return false;
  }
  value = static_cast<T>(i);
  return true;
}

inline bool
convert_value(const std::string &s, std::int8_t &value)
{
  return range_checked_convert_value<std::int32_t>(s, value);
}

inline bool
convert_value(const std::string &s, std::uint8_t &value)
{
  return range_checked_convert_value<std::uint32_t>(s, value);
}

template <typename T>
T
get_property(const properties &props, const std::string &property,
    const std::string &error_type, T default_value)
{
  static_assert(!std::is_same<T, char>::value, "char conversion is ambiguous");
  if (auto found = props.find(property); found != props.end()) {
    auto value = T{};
    if (!convert_value(found->second, value)) {
      throw std::runtime_error(error_type + " : " + property
                               + " cannot be converted from '"s + found->second
                               + "' to a type of "s + get_typename(T{}));
    }
    return value;
  }
  return default_value;
}

template <typename T>
std::optional<T>
get_property(const properties &props, const std::string &property,
    const std::string &error_type, std::optional<T> default_value)
{
  static_assert(!std::is_same<T, char>::value, "char conversion is ambiguous");
  if (auto found = props.find(property); found != props.end()) {
    auto value = T{};
    if (!convert_value(found->second, value)) {
      throw std::runtime_error(error_type + " : " + property
                               + " cannot be converted from '"s + found->second
                               + "' to a type of "s + get_typename(T{}));
    }
    return value;
  }
  return default_value;
}

template <typename T>
std::vector<T>
get_property(const properties &props, const std::string &property,
    const std::string &error_type, const std::vector<T> &default_value)
{
  if (auto found = props.find(property); found != props.end()) {
    std::vector<T> result;
    std::stringstream prop_ss(found->second);
    std::string element;
    while (getline(prop_ss, element, ',')) {
      auto value = T{};
      if (!convert_value(element, value)) {
        throw std::runtime_error(error_type + " : " + property + " cannot be converted from '"s
                                 + found->second + "' to a type of std::vector<"s
                                 + get_typename(T{}) + ">");
      }
      result.push_back(value);
    }
    return result;
  }
  return default_value;
}

template <typename T>
std::vector<std::vector<T>>
get_property(const properties &props, const std::string &property,
    const std::string &error_type, const std::vector<std::vector<T>> &default_value)
{
  if (auto found = props.find(property); found != props.end()) {
    std::vector<std::vector<T>> all_results;
    std::stringstream prop_ss(found->second);
    std::string elements;
    while (getline(prop_ss, elements, '|')) {
      std::vector<T> result;
      std::stringstream el(elements);
      std::string element;
      while (getline(el, element, ',')) {
        auto value = T{};
        if (!convert_value(element, value)) {
          throw std::runtime_error(error_type + " : " + property + " cannot be converted from '"s
                                   + found->second + "' to a type of std::vector<std::vector<"s
                                   + get_typename(T{}) + ">>");
        }
        result.push_back(value);
      }
      all_results.push_back(result);
    }
    return all_results;
  }
  return default_value;
}

} // namespace Ax
