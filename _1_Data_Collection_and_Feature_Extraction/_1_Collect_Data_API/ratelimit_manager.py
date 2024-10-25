import time


class RateLimiter:

    def __init__(self):
        self.endpoints = ["user_timeline", "users", "tweets", "following", "followers"]
        self.use_count = {
            "user_timeline": 900,
            "users": 900,
            "tweets": 900,
            "following": 15,
            "followers": 15
        }
        self.current_count = {
            "user_timeline": 0,
            "users": 0,
            "tweets": 0,
            "following": 0,
            "followers": 0
        }
        self.per_seconds = {
            "user_timeline": 900,
            "users": 900,
            "tweets": 900,
            "following": 900,
            "followers": 900
        }
        self.first_use = {
            "user_timeline": None,
            "users": None,
            "tweets": None,
            "following": None,
            "followers": None
        }
        self.last_limit = 1

    def check(self, endpoint: str):
        """
        Checks the ratelimit of the appropriate endpoint.

        Parameters
        ----------
        * endpoint: String value of the endpoint to be checked. Accepted endpoints:
          * user_timeline
          * users
          * tweets

        Returns
        -------
        * num_wait: The amount of seconds to wait before calling the function again
        """

        if endpoint not in self.endpoints:
            raise Exception(f"Endpoint {endpoint} does not exist")

        curr = time.time()

        if self.first_use[endpoint] == None:
            self.first_use[endpoint] = curr
            self.current_count[endpoint] += 1
            return 0

        if (self.current_count[endpoint] >= self.use_count[endpoint]):
            diff = curr - self.first_use[endpoint]
            if (diff <= self.per_seconds[endpoint]):
                print(
                    f"{endpoint} waiting for {self.per_seconds[endpoint] - diff} seconds.")
                time.sleep(self.per_seconds[endpoint] - diff)
                self.current_count[endpoint] = 0
                return 1
            else:
                self.first_use[endpoint] = curr
                self.current_count[endpoint] = 1
                self.last_limit = 1
                return 0

    def too_many_requests(self, endpoint: str):
        """
        Checks the ratelimit of the appropriate endpoint.

        Parameters
        ----------
        * endpoint: String value of the endpoint to be checked. RAW ENDPOINT, not edited

        """
        endpoint_code = ""
        for x in self.endpoints:
            if x in endpoint:
                errorstr = "Ratelimit_manager stats:\n"
                errorstr += f"Use count: {self.use_count[x]}\n"
                errorstr += f"Curr count: {self.current_count[x]}\n"
                errorstr += f"Per seconds: {self.per_seconds[x]}\n"
                errorstr += f"First use: {self.first_use[x]}\n"
                print(errorstr)

        print(f"Waiting {self.last_limit} before trying again...")
        time.sleep(self.last_limit)
        self.last_limit = self.last_limit * 2

        # raise Exception("Endpoint not found in appropriate list: ", self.endpoints)

    def internal_error(self):
        """
        Waits 10 seconds after internal error message sent
        """
        print("Internal error detected. Waiting 10 seconds.")
        time.sleep(10)
