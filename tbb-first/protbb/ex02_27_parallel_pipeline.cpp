#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tbb/tbb.h>

using CaseStringPtr = std::shared_ptr<std::string>;

CaseStringPtr getCaseString(std::ofstream &f);

void writeCaseString(std::ofstream &f, CaseStringPtr s);


void fig_2_27(int num_tokens, std::ofstream &caseBeforeFile, std::ofstream &caseAfterFile) {

  tbb::parallel_pipeline(
    num_tokens,

    tbb::make_filter<void, CaseStringPtr>(
      tbb::filter::serial_in_order,
      [&](tbb::flow_control &fc) -> CaseStringPtr {
        CaseStringPtr s_ptr = getCaseString(caseBeforeFile);
        if (!s_ptr)
          fc.stop();
        return s_ptr;
      }) & // concatenation operation

    /* make the change case filter */
    tbb::make_filter<CaseStringPtr, CaseStringPtr>(
      tbb::filter::parallel,
      [](CaseStringPtr s_ptr) -> CaseStringPtr {
        std::transform(
          s_ptr->begin(), s_ptr->end(), s_ptr->begin(),
          [](char c) -> char {
                         if (std::islower(c))
                           return std::toupper(c);
                         else if (std::isupper(c))
                           return std::tolower(c);
                         else
                           return c;
                       });
        return s_ptr;
      }) & // concatenation operation

    /* make the write filter */
    tbb::make_filter<CaseStringPtr, void>(
      tbb::filter::serial_in_order,
      [&](CaseStringPtr s_ptr) -> void {
        writeCaseString(caseAfterFile, s_ptr);
      })
  );
}

using CaseStringPtr = std::shared_ptr<std::string>;
static tbb::concurrent_queue<CaseStringPtr> caseFreeList;
static int numCaseInputs = 0;

void initCaseChange(int num_strings, int string_len, int free_list_size) {
  numCaseInputs = num_strings;
  caseFreeList.clear();
  for (int i = 0; i < free_list_size; ++i) {
    caseFreeList.push(std::make_shared<std::string>(string_len, ' '));
  }
}

CaseStringPtr getCaseString(std::ofstream &f) {
  std::shared_ptr<std::string> s;
  if (numCaseInputs > 0) {
    if (!caseFreeList.try_pop(s) || !s) {
      std::cerr << "Error: Ran out of elements in free list!" << std::endl;
      return CaseStringPtr{};
    }
    int ascii_range = 'z' - 'A' + 2;
    for (int i = 0; i < s->size(); ++i) {
      int offset = i % ascii_range;
      if (offset)
        (*s)[i] = 'A' + offset - 1;
      else
        (*s)[i] = '\n';
    }
    if (f.good()) {
      f << *s;
    }
    --numCaseInputs;
  }
  return s;
}

void writeCaseString(std::ofstream &f, CaseStringPtr s) {
  if (f.good()) {
    f << *s;
  }
  caseFreeList.push(s);
}

static void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.01);
  });
}

int main() {
  int num_tokens = tbb::task_scheduler_init::default_num_threads();
  int num_strings = 100;
  int string_len = 100000;
  int free_list_size = num_tokens;

  std::ofstream caseBeforeFile("fig_2_27_before.txt");
  std::ofstream caseAfterFile("fig_2_27_after.txt");
  initCaseChange(num_strings, string_len, free_list_size);

  warmupTBB();
  double parallel_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    fig_2_27(num_tokens, caseBeforeFile, caseAfterFile);
    parallel_time = (tbb::tick_count::now() - t0).seconds();
  }
  std::cout << "parallel_time == " << parallel_time << " seconds" << std::endl;
  return 0;
}