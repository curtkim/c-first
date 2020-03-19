//
// Created by curt on 20. 3. 16..
//

#ifndef MYPROJECT_EXAMPLECLASS_HPP
#define MYPROJECT_EXAMPLECLASS_HPP

class ExampleClass {

public:
  /// Create an ExampleClass
  ExampleClass();

  /// Create an ExampleClass with lot's of intial values
  ExampleClass(int a, float b);

  ~ExampleClass();

  /// This method does something
  void DoSomething();

  /** This is a method that does so
   * much that I must write an epic
   * novel just to describe how much
   * it truly does. */
  void DoNothing();

  /** \brief 유용한 메소드
   * \param level an integer setting how useful to be
   * \return Output that is extra useful
   *
   * This method does unbelievably useful things.
   * And returns exceptionally useful results.
   * Use it everyday with good health.
   */
  void *VeryUsefulMethod(bool level);

private:
  const char *fQuestion; ///< the question
  int fAnswer;           ///< the answer

  int a;
};

#endif // MYPROJECT_EXAMPLECLASS_HPP
