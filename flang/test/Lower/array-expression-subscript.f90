! RUN: bbc --emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1a(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>) {
! CHECK:         %[[VAL_3:.*]] = constant 10 : index
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 20 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_0]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_8:.*]] = constant 10 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = constant 1 : index
! CHECK:         %[[VAL_11:.*]] = constant 1 : i64
! CHECK:         %[[VAL_12:.*]] = constant 20 : i64
! CHECK:         %[[VAL_13:.*]] = constant 2 : i64
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.slice %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : (i64, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_2]](%[[VAL_14]]) {{\[}}%[[VAL_15]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<20xi32>
! CHECK:         %[[VAL_17:.*]] = constant 0 : index
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_21:.*]] = subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:         %[[VAL_22:.*]] = addi %[[VAL_21]], %[[VAL_20]] : index
! CHECK:         %[[VAL_23:.*]] = divi_signed %[[VAL_22]], %[[VAL_20]] : index
! CHECK:         %[[VAL_24:.*]] = cmpi sgt, %[[VAL_23]], %[[VAL_17]] : index
! CHECK:         %[[VAL_25:.*]] = select %[[VAL_24]], %[[VAL_23]], %[[VAL_17]] : index
! CHECK:         %[[VAL_26:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_27:.*]] = fir.slice %[[VAL_10]], %[[VAL_25]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_1]](%[[VAL_26]]) {{\[}}%[[VAL_27]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_29:.*]] = constant 1 : index
! CHECK:         %[[VAL_30:.*]] = constant 0 : index
! CHECK:         %[[VAL_31:.*]] = subi %[[VAL_9]], %[[VAL_29]] : index
! CHECK:         %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_31]] step %[[VAL_29]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_7]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_35:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_33]] : (!fir.array<20xi32>, index) -> i32
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
! CHECK:           %[[VAL_37:.*]] = subi %[[VAL_36]], %[[VAL_10]] : index
! CHECK:           %[[VAL_38:.*]] = fir.array_fetch %[[VAL_28]], %[[VAL_37]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_39:.*]] = fir.array_update %[[VAL_34]], %[[VAL_38]], %[[VAL_33]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_39]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_7]], %[[VAL_40:.*]] to %[[VAL_0]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return
! CHECK:       }
subroutine test1a(a,b,c)
  integer :: a(10), b(10), c(20)

  a = b(c(1:20:2))
end subroutine test1a

! CHECK-LABEL: func @_QPtest1b(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>) {
! CHECK:         %[[VAL_3:.*]] = constant 10 : index
! CHECK:         %[[VAL_4:.*]] = constant 10 : index
! CHECK:         %[[VAL_5:.*]] = constant 20 : index
! CHECK:         %[[VAL_6:.*]] = constant 1 : index
! CHECK:         %[[VAL_7:.*]] = constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = constant 20 : i64
! CHECK:         %[[VAL_9:.*]] = constant 2 : i64
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.slice %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : (i64, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_2]](%[[VAL_10]]) {{\[}}%[[VAL_11]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<20xi32>
! CHECK:         %[[VAL_13:.*]] = constant 0 : index
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_17:.*]] = subi %[[VAL_15]], %[[VAL_14]] : index
! CHECK:         %[[VAL_18:.*]] = addi %[[VAL_17]], %[[VAL_16]] : index
! CHECK:         %[[VAL_19:.*]] = divi_signed %[[VAL_18]], %[[VAL_16]] : index
! CHECK:         %[[VAL_20:.*]] = cmpi sgt, %[[VAL_19]], %[[VAL_13]] : index
! CHECK:         %[[VAL_21:.*]] = select %[[VAL_20]], %[[VAL_19]], %[[VAL_13]] : index
! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_23:.*]] = fir.slice %[[VAL_6]], %[[VAL_21]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_1]](%[[VAL_22]]) {{\[}}%[[VAL_23]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_25:.*]] = constant 10 : i64
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_0]](%[[VAL_27]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_29:.*]] = constant 1 : index
! CHECK:         %[[VAL_30:.*]] = constant 0 : index
! CHECK:         %[[VAL_31:.*]] = subi %[[VAL_26]], %[[VAL_29]] : index
! CHECK:         %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_31]] step %[[VAL_29]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_24]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_35:.*]] = fir.array_fetch %[[VAL_28]], %[[VAL_33]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_36:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_33]] : (!fir.array<20xi32>, index) -> i32
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
! CHECK:           %[[VAL_38:.*]] = subi %[[VAL_37]], %[[VAL_6]] : index
! CHECK:           %[[VAL_39:.*]] = fir.array_update %[[VAL_34]], %[[VAL_35]], %[[VAL_38]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_39]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_24]], %[[VAL_40:.*]] to %[[VAL_1]]{{\[}}%[[VAL_23]]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }
subroutine test1b(a,b,c)
  integer :: a(10), b(10), c(20)

  b(c(1:20:2)) = a
end subroutine test1b

subroutine test1c(a,b,c,d)
  integer :: a(10), b(10), d(10), c(20)

  ! flang: parser FAIL (final position)
  !a = b(d(c(1:20:2))
end subroutine test1c
