! RUN: bbc --emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest1a(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>) {
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = fir.array_load %[[VAL_0]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_8:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_12:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_13:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.slice %[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : (i64, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_16:.*]] = fir.array_load %[[VAL_2]](%[[VAL_14]]) {{\[}}%[[VAL_15]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<20xi32>
! CHECK:         %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_21:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:         %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_20]] : index
! CHECK:         %[[VAL_23:.*]] = arith.divsi %[[VAL_22]], %[[VAL_20]] : index
! CHECK:         %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_17]] : index
! CHECK:         %[[VAL_25:.*]] = select %[[VAL_24]], %[[VAL_23]], %[[VAL_17]] : index
! CHECK:         %[[VAL_26:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_27:.*]] = fir.slice %[[VAL_10]], %[[VAL_25]], %[[VAL_10]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_1]](%[[VAL_26]]) {{\[}}%[[VAL_27]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_31:.*]] = arith.subi %[[VAL_9]], %[[VAL_29]] : index
! CHECK:         %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_31]] step %[[VAL_29]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_7]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_35:.*]] = fir.array_fetch %[[VAL_16]], %[[VAL_33]] : (!fir.array<20xi32>, index) -> i32
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_36]], %[[VAL_10]] : index
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
! CHECK:         %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.slice %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : (i64, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_2]](%[[VAL_10]]) {{\[}}%[[VAL_11]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<20xi32>
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_17:.*]] = arith.subi %[[VAL_15]], %[[VAL_14]] : index
! CHECK:         %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_16]] : index
! CHECK:         %[[VAL_19:.*]] = arith.divsi %[[VAL_18]], %[[VAL_16]] : index
! CHECK:         %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_13]] : index
! CHECK:         %[[VAL_21:.*]] = select %[[VAL_20]], %[[VAL_19]], %[[VAL_13]] : index
! CHECK:         %[[VAL_22:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_23:.*]] = fir.slice %[[VAL_6]], %[[VAL_21]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_24:.*]] = fir.array_load %[[VAL_1]](%[[VAL_22]]) {{\[}}%[[VAL_23]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_25:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_28:.*]] = fir.array_load %[[VAL_0]](%[[VAL_27]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_31:.*]] = arith.subi %[[VAL_26]], %[[VAL_29]] : index
! CHECK:         %[[VAL_32:.*]] = fir.do_loop %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_31]] step %[[VAL_29]] unordered iter_args(%[[VAL_34:.*]] = %[[VAL_24]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_35:.*]] = fir.array_fetch %[[VAL_28]], %[[VAL_33]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_36:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_33]] : (!fir.array<20xi32>, index) -> i32
! CHECK:           %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
! CHECK:           %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_6]] : index
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

! CHECK-LABEL: func @_QPtest1c(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<20xi32>>, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>) {
! CHECK:         return
! CHECK:       }
subroutine test1c(a,b,c,d)
  integer :: a(10), b(10), d(10), c(20)

  ! flang: parser FAIL (final position)
  !a = b(d(c(1:20:2))
end subroutine test1c


! CHECK-LABEL: func @_QPtest2a(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>) {
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_9:.*]] = fir.array_load %[[VAL_0]](%[[VAL_8]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_10:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_3]](%[[VAL_14]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_17:.*]] = fir.slice %[[VAL_13]], %[[VAL_7]], %[[VAL_13]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_2]](%[[VAL_16]]) {{\[}}%[[VAL_17]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_20:.*]] = arith.subi %[[VAL_7]], %[[VAL_13]] : index
! CHECK:         %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_13]] : index
! CHECK:         %[[VAL_22:.*]] = arith.divsi %[[VAL_21]], %[[VAL_13]] : index
! CHECK:         %[[VAL_23:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_19]] : index
! CHECK:         %[[VAL_24:.*]] = select %[[VAL_23]], %[[VAL_22]], %[[VAL_19]] : index
! CHECK:         %[[VAL_25:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_26:.*]] = fir.slice %[[VAL_12]], %[[VAL_24]], %[[VAL_12]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_27:.*]] = fir.array_load %[[VAL_1]](%[[VAL_25]]) {{\[}}%[[VAL_26]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_11]], %[[VAL_28]] : index
! CHECK:         %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_9]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_34:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_32]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (i32) -> index
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_13]] : index
! CHECK:           %[[VAL_37:.*]] = fir.array_fetch %[[VAL_18]], %[[VAL_36]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i32) -> index
! CHECK:           %[[VAL_39:.*]] = arith.subi %[[VAL_38]], %[[VAL_12]] : index
! CHECK:           %[[VAL_40:.*]] = fir.array_fetch %[[VAL_27]], %[[VAL_39]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_41:.*]] = fir.array_update %[[VAL_33]], %[[VAL_40]], %[[VAL_32]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_41]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_9]], %[[VAL_42:.*]] to %[[VAL_0]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>
! CHECK:         return
! CHECK:       }
subroutine test2a(a,b,c,d)
  integer :: a(10), b(10), c(10), d(10)

  a = b(c(d))
end subroutine test2a

! CHECK-LABEL: func @_QPtest2b(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>>, %[[VAL_3:.*]]: !fir.ref<!fir.array<10xi32>>) {
! CHECK:         %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_3]](%[[VAL_10]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.slice %[[VAL_9]], %[[VAL_7]], %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_14:.*]] = fir.array_load %[[VAL_2]](%[[VAL_12]]) {{\[}}%[[VAL_13]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[VAL_7]], %[[VAL_9]] : index
! CHECK:         %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_9]] : index
! CHECK:         %[[VAL_18:.*]] = arith.divsi %[[VAL_17]], %[[VAL_9]] : index
! CHECK:         %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_15]] : index
! CHECK:         %[[VAL_20:.*]] = select %[[VAL_19]], %[[VAL_18]], %[[VAL_15]] : index
! CHECK:         %[[VAL_21:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_22:.*]] = fir.slice %[[VAL_8]], %[[VAL_20]], %[[VAL_8]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_23:.*]] = fir.array_load %[[VAL_1]](%[[VAL_21]]) {{\[}}%[[VAL_22]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_24:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i64) -> index
! CHECK:         %[[VAL_26:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_27:.*]] = fir.array_load %[[VAL_0]](%[[VAL_26]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.array<10xi32>
! CHECK:         %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_29:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_25]], %[[VAL_28]] : index
! CHECK:         %[[VAL_31:.*]] = fir.do_loop %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_30]] step %[[VAL_28]] unordered iter_args(%[[VAL_33:.*]] = %[[VAL_23]]) -> (!fir.array<10xi32>) {
! CHECK:           %[[VAL_34:.*]] = fir.array_fetch %[[VAL_27]], %[[VAL_32]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_35:.*]] = fir.array_fetch %[[VAL_11]], %[[VAL_32]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_36:.*]] = fir.convert %[[VAL_35]] : (i32) -> index
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_36]], %[[VAL_9]] : index
! CHECK:           %[[VAL_38:.*]] = fir.array_fetch %[[VAL_14]], %[[VAL_37]] : (!fir.array<10xi32>, index) -> i32
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> index
! CHECK:           %[[VAL_40:.*]] = arith.subi %[[VAL_39]], %[[VAL_8]] : index
! CHECK:           %[[VAL_41:.*]] = fir.array_update %[[VAL_33]], %[[VAL_34]], %[[VAL_40]] : (!fir.array<10xi32>, i32, index) -> !fir.array<10xi32>
! CHECK:           fir.result %[[VAL_41]] : !fir.array<10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_23]], %[[VAL_42:.*]] to %[[VAL_1]]{{\[}}%[[VAL_22]]] : !fir.array<10xi32>, !fir.array<10xi32>, !fir.ref<!fir.array<10xi32>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }
subroutine test2b(a,b,c,d)
  integer :: a(10), b(10), c(10), d(10)

  b(c(d)) = a
end subroutine test2b
