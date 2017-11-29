/*
 * File: semaphore4.c
 *
 *
 * --------------------------------------------------------------------------
 *
 *      Pthreads-win32 - POSIX Threads Library for Win32
 *      Copyright(C) 1998 John E. Bossom
 *      Copyright(C) 1999,2012 Pthreads-win32 contributors
 *
 *      Homepage1: http://sourceware.org/pthreads-win32/
 *      Homepage2: http://sourceforge.net/projects/pthreads4w/
 *
 *      The current list of contributors is contained
 *      in the file CONTRIBUTORS included with the source
 *      code distribution. The list can also be seen at the
 *      following World Wide Web location:
 *      http://sources.redhat.com/pthreads-win32/contributors.html
 *
 *      This library is free software; you can redistribute it and/or
 *      modify it under the terms of the GNU Lesser General Public
 *      License as published by the Free Software Foundation; either
 *      version 2 of the License, or (at your option) any later version.
 *
 *      This library is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *      Lesser General Public License for more details.
 *
 *      You should have received a copy of the GNU Lesser General Public
 *      License along with this library in the file COPYING.LIB;
 *      if not, write to the Free Software Foundation, Inc.,
 *      59 Temple Place - Suite 330, Boston, MA 02111-1307, USA
 *
 * --------------------------------------------------------------------------
 *
 * Test Synopsis: Verify sem_getvalue returns the correct number of waiters
 * after threads are cancelled.
 * -
 *
 * Test Method (Validation or Falsification):
 * - Validation
 *
 * Requirements Tested:
 * -
 *
 * Features Tested:
 * -
 *
 * Cases Tested:
 * -
 *
 * Description:
 * -
 *
 * Environment:
 * -
 *
 * Input:
 * - None.
 *
 * Output:
 * - File name, Line number, and failed expression on failure.
 * - No output on success.
 *
 * Assumptions:
 * -
 *
 * Pass Criteria:
 * - Process returns zero exit status.
 *
 * Fail Criteria:
 * - Process returns non-zero exit status.
 */

#include "test.h"

#define MAX_COUNT 100

static sem_t s;

static void *
thr (void * arg)
{
  assert(sem_wait(&s) == 0);
  return NULL;
}

#ifndef MONOLITHIC_PTHREAD_TESTS
int
main()
#else
int
test_semaphore4(void)
#endif
{
	int value = 0;
	int i;
	pthread_t t[MAX_COUNT+1];

	assert(sem_init(&s, PTHREAD_PROCESS_PRIVATE, 0) == 0);
	assert(sem_getvalue(&s, &value) == 0);
	assert(value == 0);

	for (i = 1; i <= MAX_COUNT; i++)
		{
			assert(pthread_create(&t[i], NULL, thr, NULL) == 0);
			do {
			  sched_yield();
			  assert(sem_getvalue(&s, &value) == 0);
			} while (value != -i);
			assert(-value == i);
		}

	assert(sem_getvalue(&s, &value) == 0);
	assert(-value == MAX_COUNT);
	assert(pthread_cancel(t[50]) == 0);
	  {
	    void* result;
	    assert(pthread_join(t[50], &result) == 0);
	  }
	assert(sem_getvalue(&s, &value) == 0);
	assert(-value == (MAX_COUNT - 1));

	for (i = MAX_COUNT - 2; i >= 0; i--)
		{
			assert(sem_post(&s) == 0);
			assert(sem_getvalue(&s, &value) == 0);
			assert(-value == i);
		}

	for (i = 1; i <= MAX_COUNT; i++)
	  if (i != 50)
	    assert(pthread_join(t[i], NULL) == 0);

  return 0;
}

