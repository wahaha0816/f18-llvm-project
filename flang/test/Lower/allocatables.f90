! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of allocatables
! CHECK-LABEL: _QPfoo
subroutine foo()
  real, allocatable :: x(:), y(:, :), z
  ! CHECK: %[[xBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {name = "_QFfooEx"}
  ! CHECK-DAG: %[[xNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK-DAG: %[[xNullShape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[xInitEmbox:.*]] = fir.embox %[[xNullAddr]](%[[xNullShape]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK: fir.store %[[xInitEmbox]] to %[[xBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>

  ! CHECK: %[[yBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>> {name = "_QFfooEy"}
  ! CHECK-DAG: %[[yNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x?xf32>>
  ! CHECK-DAG: %[[yNullShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[yInitEmbox:.*]] = fir.embox %[[yNullAddr]](%[[yNullShape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK: fir.store %[[yInitEmbox]] to %[[yBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>

  ! CHECK: %[[zBoxAddr:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {name = "_QFfooEz"}
  ! CHECK: %[[zNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<f32>
  ! CHECK: %[[zInitEmbox:.*]] = fir.embox %[[zNullAddr]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  ! CHECK: fir.store %[[zInitEmbox]] to %[[zBoxAddr]] : !fir.ref<!fir.box<!fir.heap<f32>>>


  allocate(x(42:100), y(43:50, 51), z)
  ! CHECK-DAG: %[[xlb:.*]] = constant 42 : i32
  ! CHECK-DAG: %[[xub:.*]] = constant 100 : i32
  ! CHECK-DAG: %[[xBoxCast2:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[xlbCast:.*]] = fir.convert %[[xlb]] : (i32) -> i64
  ! CHECK-DAG: %[[xubCast:.*]] = fir.convert %[[xub]] : (i32) -> i64
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds(%[[xBoxCast2]], %c0{{.*}}, %[[xlbCast]], %[[xubCast]]) : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
  ! CHECK-DAG: %[[xBoxCast3:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[errMsg:.*]] = fir.convert %{{.*}} : (!fir.ref<none>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[sourceFile:.*]] = fir.convert %{{.*}} -> !fir.ref<i8>
  ! CHECK: fir.call @{{.*}}AllocatableAllocate(%[[xBoxCast3]], %false{{.*}}, %[[errMsg]], %[[sourceFile]], %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.ref<!fir.box<none>>, !fir.ref<i8>, i32) -> i32

  ! Simply check that we are emitting the right numebr of set bound for y and z. Otherwise, this is just like x.
  ! CHECK: fir.convert %[[yBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate
  ! CHECK: %[[zBoxCast:.*]] = fir.convert %[[zBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-NOT: fir.call @{{.*}}AllocatableSetBounds
  ! CHECK: fir.call @{{.*}}AllocatableAllocate

  ! Check that y descriptor is read when referencing it.
  ! CHECK: %[[yBoxLoad:.*]] = fir.load %[[yBoxAddr]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: %[[yAddr:.*]] = fir.box_addr %[[yBoxLoad]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
  ! CHECK: %[[yBounds1:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK: %[[yBounds2:.*]]:3 = fir.box_dims %[[yBoxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  print *, x, y(45, 46), z

  deallocate(x, y, z)
  ! CHECK: %[[xBoxCast4:.*]] = fir.convert %[[xBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[xBoxCast4]], {{.*}})
  ! CHECK: %[[yBoxCast4:.*]] = fir.convert %[[yBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[yBoxCast4]], {{.*}})
  ! CHECK: %[[zBoxCast4:.*]] = fir.convert %[[zBoxAddr]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}AllocatableDeallocate(%[[zBoxCast4]], {{.*}})
end subroutine

! CHECK-LABEL: func @_QPtest_globals()
subroutine test_globals()
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgx) : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK-DAG: fir.address_of(@_QFtest_globalsEgy) : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  integer, allocatable :: gx, gy(:, :)
  save :: gx, gy
  allocate(gx, gy(20, 30))
end subroutine

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgx : !fir.box<!fir.heap<i32>>
  ! CHECK: %[[gxNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<i32>
  ! CHECK: %[[gxInitBox:.*]] = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK: fir.has_value %[[gxInitBox]] : !fir.box<!fir.heap<i32>>

! CHECK-LABEL: fir.global internal @_QFtest_globalsEgy : !fir.box<!fir.heap<!fir.array<?x?xi32>>> {
  ! CHECK-DAG: %[[gyNullAddr:.*]] = fir.convert %c0{{.*}} : (index) -> !fir.heap<!fir.array<?x?xi32>>
  ! CHECK-DAG: %[[gyShape:.*]] = fir.shape %c0{{.*}}, %c0{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK: %[[gyInitBox:.*]] = fir.embox %[[gyNullAddr]](%[[gyShape]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK: fir.has_value %[[gyInitBox]] : !fir.box<!fir.heap<!fir.array<?x?xi32>>>

