#include <cppcoro/sync_wait.hpp>

#include <cppcoro/config.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/shared_task.hpp>
#include <cppcoro/on_scope_exit.hpp>
#include <cppcoro/static_thread_pool.hpp>

#include <string>
#include <type_traits>

int main()
{
	auto makeTask = []() -> cppcoro::task<std::string>
	{
		co_return "foo";
	};

	auto task = makeTask();
	assert(cppcoro::sync_wait(task) == "foo");

    return 0;
}
