#include <iostream>

#include <cassert>
#include <memory>

template <class T>
struct list {
    struct node {
        T data{};
        std::unique_ptr<node> next{};
    };

    ~list() noexcept(true) { clean(); }

    auto push_back(T&& data) {
        auto tmp = std::make_unique<node>(std::move(data));
        tmp->next = std::move(head);
        head = std::move(tmp);
    }

    auto pop() {
        if (not head) {
            return;
        }
        auto tmp = std::move(head);
        head = std::move(tmp->next);
    }

    [[nodiscard]] auto size() const -> std::size_t {
        std::size_t size{};
        for (auto tmp = head.get(); tmp; tmp = tmp->next.get()) {
            ++size;
        }
        return size;
    }

    [[nodiscard]] auto front() const -> const T& {
        assert(head.get());
        return head->data;
    }

    auto clean() -> void {
        while (head) {
            head.reset(head->next.release());
        }
    }

private:
    std::unique_ptr<node> head{};
};

#include <tuple>

int main() {
    static_assert(
            0u ==
            [] {
                list<int> list{};
                list.push_back(42);
                list.clean();
                return list.size();
            }(),
            "clean");

    static_assert(
            0u ==
            [] {
                list<int> list{};
                return list.size();
            }(),
            "size");

    static_assert(
            2u ==
            [] {
                list<int> list{};
                list.push_back(42);
                list.push_back(43);
                return list.size();
            }(),
            "size many");

    static_assert(
            3u ==
            [] {
                list<int> list{};
                list.push_back(42);
                list.push_back(43);
                list.push_back(44);
                return list.size();
            }(),
            "size many");

    static_assert(
            1u ==
            [] {
                list<int> list{};
                list.push_back(42);
                return list.size();
            }(),
            "push_back");

    static_assert(
            42 ==
            [] {
                list<int> list{};
                list.push_back(42);
                return list.front();
            }(),
            "front");

    static_assert(
            std::tuple{2u, 43} ==
            [] {
                list<int> list{};
                list.push_back(42);
                list.push_back(43);
                return std::tuple{list.size(), list.front()};
            }(),
            "front many");

    static_assert(
            0u ==
            [] {
                list<int> list{};
                list.pop();
                return list.size();
            }(),
            "pop empty");

    static_assert(
            0u ==
            [] {
                list<int> list{};
                list.push_back(42);
                list.pop();
                return list.size();
            }(),
            "pop");
}
