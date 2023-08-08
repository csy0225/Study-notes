# op注册机制

## op注册关键代码
```
#define REGISTER_LITE_OP(op_type__, OpClass)                                   \
  static paddle::lite::OpLiteRegistrar op_type__##__registry(                  \
      #op_type__, []() {                                                       \
        return std::unique_ptr<paddle::lite::OpLite>(new OpClass(#op_type__)); \
      });                                                                      \
  int touch_op_##op_type__() {                                                 \
    op_type__##__registry.touch();                                             \
    OpKernelInfoCollector::Global().AddOp2path(#op_type__, __FILE__);          \
    return 0;                                                                  \
  }
```
作用：注册算子，例如 REGISTER_LITE_OP(conv2d, paddle::lite::operators::ConvOpLite);。
内容：
  （1）通过宏定义了一个paddle::lite::OpLiteRegistrar 类型的变量 op_type__##__registry, 并用static关键字进行了修饰，例如 conv2d__registry。
  （2）通过宏定义了一个 touch_op_##op_type__（）函数，例如 
    ```
    touch_op_conv2d() {
      conv2d__registry.touch(); // 确保conv2d__registry已经被初始化过了。
      OpKernelInfoCollector::Global().AddOp2path(#op_type__, __FILE__);
      return 0; 
    }
    ```
## 源代码解读
```
class OpLiteFactory {
 public:
  // Register a function to create an op
  void RegisterCreator(const std::string& op_type,
                       std::function<std::shared_ptr<OpLite>()> fun) {
    op_registry_[op_type] = fun;
  }

  static OpLiteFactory& Global() {
    static OpLiteFactory* x = new OpLiteFactory;
    return *x;
  }
  #  Global函数：该函数使用new在堆上创建了一片内存空间，然后把这段内存空间的地址赋给了x，x本来是局部变量应该存储在栈上，但是受到了static关键字的修饰，所以x会存储在静态变量区。最终函数会返回x指向的那一片内存的引用。

  std::shared_ptr<OpLite> Create(const std::string& op_type) const {
    auto it = op_registry_.find(op_type);
    if (it == op_registry_.end()) return nullptr;
    return it->second();
  }

  std::string DebugString() const {
    STL::stringstream ss;
    for (const auto& item : op_registry_) {
      ss << " - " << item.first << "\n";
    }
    return ss.str();
  }

  std::vector<std::string> GetAllOps() const {
    std::vector<std::string> res;
    for (const auto& op : op_registry_) {
      res.push_back(op.first);
    }
    return res;
  }

 protected:
  std::map<std::string, std::function<std::shared_ptr<OpLite>()>> op_registry_;
};
```

### 涉及的知识点
1. 设计模式：工厂模式

