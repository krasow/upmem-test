cd libs
# upmem runtime
# wget http://sdk-releases.upmem.com/2024.1.0/ubuntu_22.04/upmem-2024.1.0-Linux-x86_64.tar.gz
# wget http://sdk-releases.upmem.com/2024.1.0/rocky_8/upmem-2024.1.0-Linux-x86_64.tar.gz
# wget http://sdk-releases.upmem.com/2023.2.0/centos_7/upmem-2023.2.0-Linux-x86_64.tar.gz

# for this machine, debian
wget https://cheesemonger.cs.northwestern.edu/upmem/software/upmem-2025.1.0-Linux-x86_64.tar.gz

tar -xf upmem-2025.1.0-Linux-x86_64.tar.gz
rm upmem-2025.1.0-Linux-x86_64.tar.gz

# patch upmem runtime
cat > ./upmem-2025.1.0-Linux-x86_64/share/upmem/include/stdlib/stddef.h <<EOF
/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 *
 * Modified by David Krasowska krasow@u.northwestern.edu to change types to support C++ 
 * compilation for the Legion runtime system.
 */

#ifndef _DPUSYSCORE_STDDEF_H_
#define _DPUSYSCORE_STDDEF_H_

/**
 * @file stddef.h
 * @brief Defines some standard types.
 */

/**
 * @brief The type for the result of subtracting two pointers.
 */
typedef int ptrdiff_t;

/**
 * @brief The type for the result of the <code>sizeof</code> operator.
 *
 * @internal No memory or object in the DPU can be more than megabytes of data.
 *           As a consequence, we can restrict the sizes to 32 bits, which is
 *           already a lot.
 */
typedef unsigned int size_t; // change for legion
/**
 * @brief Value whose alignment requirement is at least as strict (as large) as that of every scalar type.
 */
typedef unsigned long int max_align_t;

#ifndef __cplusplus // compiling with c++ for legion
/**
 * @brief The type for wide-character codes.
 */
typedef unsigned int wchar_t;
/**
 * @def NULL
 * @brief The null pointer constant.
 */
#define NULL ((void *)0) 
#else
#define NULL ((char *)0)
#endif

/**
 * @def offsetof
 * @hideinitializer
 * @brief Offset in bytes to the structure member, from the beginning of its structure.
 *
 * @param st the structure
 * @param m the member name
 *
 * @internal Raw version of offsetof, should be enough in our context, with all the underlying risks.
 */
#define offsetof(st, m) ((size_t)(&((st *)0)->m))

#endif /* _DPUSYSCORE_STDDEF_H_ */
EOF

cd ..
