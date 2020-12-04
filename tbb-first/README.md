## Reference
- https://www.threadingbuildingblocks.org/docs/help/reference/flow_graph/sender_and_buffer_policy.html
- https://nlguillemot.wordpress.com/2016/10/24/tbb-flow-graph-nodes-summary/

## 실행구조
### Case1

    graph g;
    function_node< int, int > n( g, unlimited, []( int v ) -> int {
        cout << v;
        spin_for( v );
        cout << v;
        return v;
    } );
    function_node< int, int > m( g, 1, []( int v ) -> int {
        v *= v;
        cout << v;
        spin_for( v );
        cout << v;
        return v;
    } );
    make_edge( n, m );
    n.try_put( 1 );
    n.try_put( 2 );
    n.try_put( 3 );
    g.wait_for_all();

![thread chart](https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/Images/execution_timeline2node.jpg)

### Case2

    typedef continue_node< continue_msg > node_t;
    typedef const continue_msg & msg_t;
    
    int main() {
    tbb::flow::graph g;
    node_t A(g, [](msg_t){ a(); } );
    node_t B(g, [](msg_t){ b(); } );
    node_t C(g, [](msg_t){ c(); } );
    node_t D(g, [](msg_t){ d(); } );
    node_t E(g, [](msg_t){ e(); } );
    node_t F(g, [](msg_t){ f(); } );
    make_edge(A, B);
    make_edge(B, C);
    make_edge(B, D);
    make_edge(A, E);
    make_edge(E, D);
    make_edge(E, F);
    A.try_put( continue_msg() );
    g.wait_for_all();
    return 0;
    }

![dependance graph](https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/Images/dependence_graph.jpg)
![thread chart](https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/Images/execution_timeline_dependence.jpg)
