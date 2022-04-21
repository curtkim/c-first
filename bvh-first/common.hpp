#pragma once

struct alignas(16) float3{
  float3() = default;
  float3( const float a, const float b, const float c ) : x( a ), y( b ), z( c ) {}
  float3( const float a ) : x( a ), y( a ), z( a ) {}
  union { struct { float x, y, z; float dummy; }; float cell[4]; };
  float operator [] ( const int n ) const { return cell[n]; }
};