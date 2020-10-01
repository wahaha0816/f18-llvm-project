// Types
class TrivialType {
};
class StaticLenChar {
public:
  int64_t getCharLenConst() const {return len;}
private:
  int64_t len;
};
class DynamicLenChar {
  std::variant<FromBox, Fortran::semantics::SomeExpr> len;
};
class StaticLenDerived {
private:
  llvm::SmallVector<int64_t, 8> lens;
};
class DynamicLenDerived {
private:
  llvm::SmallVector<Fortran::semantics::SomeExpr, 8> lens;
};

class Types : matcher<Types> {
  using VT = std::variant<TrivialType, StaticLenChar, DynamicLenChar, StaticLenDerived, DynamicLenDerived>;
public:
  // TODO add an isa to matcher :)
  constexpr bool isChar() const {return isa<StaticLenChar>() || isa<DynamicLenChar>()};
  constexpr bool isTrivialType() const {return isa<TrivialType>()};
  constexpr bool isDerivedType() const {return !isChar() && !isTrivialType()};
  llvm::Optional<int64_t> getCharLenConst() const {
    using RT = llvm::Optional<int64_t>;
    return match(
      [&](auto StaticLenChar& x) -> RT {return x.getCharLenConst();},
      [&](auto auto x) -> RT {return {};});
  }
  template<typename CallBack1, typename CallBack2>
  mlir::Value lowerCharLen(CallBack1 exprToMlir, CallBack2 intToMlir, Fortran::lower::SymbolBox Box /* for FromBox*/ ) {
    // TODO
  }
private:
  VT matchee;
};


// Shape
class Scalar {
};

class StaticShapeArray {
public:
  bool lboundAllOnes() const {
    return llvm::all_of(lbounds, [](int64_t v) { return v == 1; });
  }
private:
  llvm::SmallVector<int64_t, 8> lbounds;
  llvm::SmallVector<int64_t, 8> shapes;
};

class DynamicShapeArray {
public:
  bool lboundAllOnes() const {
    return llvm::all_of(bounds, [](const Fortran::semantics::ShapeSpec *p) {
        if (auto low = p->lbound().GetExplicit())
          if (auto lb = Fortran::evaluate::ToInt64(*low))
            return *lb == 1;
        return false;
      });
  }
private:
  llvm::SmallVector<const Fortran::semantics::ShapeSpec *, 8> bounds;
};

class Shapes : matcher<Shapes> {
  using VT = std::variant<Scalar, StaticShapeArray, DynamicShapeArray>;
public:
  constexpr bool isScalar() const {return shape.isa<Scalar>();};
  constexpr bool isStaticArray() const {return shape.isa<StaticShapeArray>();};
  bool lboundAllOnes() const {
    return match(
      [&](auto StaticShapeArray& x) -> RT {return x.lboundAllOnes();},
      [&](auto DynamicShapeArray& x) -> RT {return x.lboundAllOnes();},
      [&](auto auto x) -> RT {llvm::report_fatal_error("not an array");});
  }
  template<typename CallBack1, typename CallBack2>
  void lowerArrayShape(llvm::SmallVectorImpl<mlir::Value> lbounds, llvm::SmallVectorImpl<mlir::Value> extents, CallBack2 exprToMlir, CallBack2 intToMlir) {
    // TODO in StaticShapeArray and DynamicShapeArray from populateLBoundsExtents code.
  }
private:
  VT matchee;
};

class SymbolTypeAndShape : public Types, Shapes {
private:
  const Fortran::semantics::Symbol *sym;
};

SymbolTypeAndShape analyze(const Fortran::semantics::Symbol*) {
  // TODO
}


/// Then in mapSymbolAttributes
void mapSymbolAttributes ( /* ... */) {
  /* ... */
  auto idxTy = 
  auto exprToMlir = [&](const Fortran::evaluate::SomeExpr* expr) -> mlir::Value {
    return createFIRExpr(loc, expr);
  };
  auto intToMlir = [&](int64_t i) -> mlir::Value {
    return builder->createIntegerConstant(loc, idxTy, i)
  };

  auto sba = analyze(sym);
  if (sba.isScalar() && sba.isTrivialType()) {
    if (isDummy) {
        // This is an argument.
        if (!lookupSymbol(sym))
          mlir::emitError(loc, "symbol \"")
              << toStringRef(sym.name()) << "\" must already be in map";
        return;
      }
      // Otherwise, it's a local variable or function result.
      auto local = createNewLocal(loc, var, preAlloc);
      addSymbol(sym, local);
      return;
  }

  if (sba.Scalar() && sba.isChar()) {
    auto addr = lookupSymbol(sym);
    auto len = sba.lowerCharLen(exprToMlir, intToMlir, addr);
    if (replace) {
      // I would be in favor of moving the unbox before, but that's a different topic
      addr = charHelp.createUnboxChar(addr).getResult(0);
      addr = charHelp.createEmboxChar(addr, len);
    } else {
      auto charTy = genType(var);
      addr =
            preAlloc ? preAlloc
                     : charHelp.createCharacterTemp(charTy, charLen);
    }
    addCharSymbol(sym, local, len, replace);
    return;
  }

  if (sba.isScalar() && sba.isDerivedType()) {
    TODO("derived type scalar lowering");
  }

  if (sba.isArray() && sba.isTrivialType()) {
    llvm::SmallVector<mlir::Value, 8> extents; 
    llvm::SmallVector<mlir::Value, 8> lbounds;
    mlir::Value addr = lookupSymbol(sym);
    sba.lowerArrayShape(lbounds, extents, exprToMlir, intToMlir /* should probably pass symBox for descriptor arguments */);
    if (!addr)
      addr = createNewLocal(loc, var, preAlloc);
    if (sba.lboundAllOnes())
      localSymbols.addSymbolWithShape(sym, addr, shape, replace);
    else
      localSymbols.addSymbolWithBounds(sym, addr, extents, lbounds,
                                   replace);
    return;
  }

  if (sba.isArray() && sba.Char()) {
    auto addr = lookupSymbol(sym);
    // We should probably be carefull here with Character(*) array 
    // and divide the lenght with the array size ?
    auto len = sba.lowerCharLen(exprToMlir, intToMlir, addr);
    llvm::SmallVector<mlir::Value, 8> extents; 
    llvm::SmallVector<mlir::Value, 8> lbounds;
    sba.lowerArrayShape(lbounds, extents, exprToMlir, intToMlir /* should probably pass symBox for descriptor arguments */);
    if (!addr)
      addr = createNewLocal(loc, var, preAlloc);
    if (sba.lboundAllOnes())
      localSymbols.addCharSymbolWithShape(sym, addr, len, shape,
                                    replace);
    else
      localSymbols.addCharSymbolWithBounds(sym, addr, len, extents,
                                     lbounds, replace);
    return;
  }

  if (sba.isArray() && sba.isDerivedType()) {
    TODO("derived type scalar lowering");
  }

  TODO("Assumed type and rank");
}
